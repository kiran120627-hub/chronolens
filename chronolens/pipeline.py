"""End-to-end analysis: perception -> re-identification -> events -> (captions) -> annotated video.

Results live in analysis/<video-hash>/ and are reused, so a video is only processed once:
  meta.json · settings.json · detections.csv · series.csv · appearance.npz · thumbs/
  identities.csv · reid_links.csv · events.csv · segments.json · annotated.mp4
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import Settings, fmt_t
from .events import build_events
from .perception import Progress, run_perception
from .reid import build_tracklets, link_tracklets

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"

PALETTE = [(66, 135, 245), (245, 66, 93), (46, 204, 113), (241, 196, 15), (155, 89, 182), (230, 126, 34),
           (26, 188, 156), (236, 64, 122), (52, 73, 94), (149, 165, 166), (211, 84, 0), (39, 174, 96)]


@dataclass
class Analysis:
    dir: Path
    meta: dict
    settings: Settings
    detections: pd.DataFrame
    identities: pd.DataFrame
    links: pd.DataFrame
    events: pd.DataFrame
    series: pd.DataFrame
    segments: dict

    @property
    def video(self) -> Path:
        return Path(self.meta["path"])

    @property
    def annotated(self) -> Path:
        return self.dir / "annotated.mp4"

    def thumb(self, gid: str) -> Path | None:
        row = self.identities[self.identities.id == gid]
        if row.empty:
            return None
        p = self.dir / "thumbs" / f"track_{int(row.iloc[0].thumb_track)}.jpg"
        return p if p.exists() else None


def video_key(path: Path, settings: Settings) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(8 << 20))
    h.update(str(path.stat().st_size).encode())
    d = settings.to_dict()
    perception_keys = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                       for k, v in d.items() if k in ("target_fps", "model", "conf", "imgsz", "track_buffer",
                                                      "captions_every")}
    # only activity zones change the perception pass (motion masks); area zones are post-processing
    perception_keys["activity_zones"] = [[round(float(c), 3) for p in z["points"] for c in p]
                                         for z in d["zones"] if z["kind"] == "activity"]
    h.update(json.dumps(perception_keys, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def analyze(video: str | Path, settings: Settings, progress: Progress | None = None, force: bool = False,
            llm=None) -> Analysis:
    progress = progress or (lambda f, m: None)
    video = Path(video).resolve()
    settings = auto_tune(video, settings, progress)
    out = ANALYSIS / f"{video.stem[:40]}-{video_key(video, settings)}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(settings.to_dict(), indent=2, default=str))
    if force or not (out / "detections.csv").exists():
        run_perception(video, out, settings, lambda f, m: progress(0.05 + 0.75 * f, m))
    meta = json.loads((out / "meta.json").read_text())
    return post_process(out, meta, settings, progress, llm=llm)


def auto_tune(video: Path, settings: Settings, progress: Progress) -> Settings:
    """Long videos: lower the sampling rate so a 10-minute clip still finishes in a few minutes."""
    from dataclasses import replace

    from .perception import probe

    dur = probe(video, settings.target_fps).duration
    cap = 10.0 if dur <= 300 else 6.0 if dur <= 900 else 4.0 if dur <= 1800 else 2.0
    if settings.target_fps > cap:
        progress(0.01, f"Long video ({fmt_t(dur)}): analysing at {cap:g} samples/s instead of {settings.target_fps:g}")
        return replace(settings, target_fps=cap)
    return settings


def post_process(out: Path, meta: dict, settings: Settings, progress: Progress | None = None, llm=None) -> Analysis:
    """Everything after the GPU pass — cheap, so it re-runs whenever thresholds change."""
    progress = progress or (lambda f, m: None)
    progress(0.82, "Re-identifying people and objects across gaps")
    det = pd.read_csv(out / "detections.csv")
    series = pd.read_csv(out / "series.csv")
    app = dict(np.load(out / "appearance.npz")) if (out / "appearance.npz").exists() else {}
    tracklets, det = build_tracklets(det, settings)
    gid_of, identities, links = link_tracklets(tracklets, app, meta["width"], meta["height"], settings)
    det["gid"] = det.track.map(gid_of)
    det = det.dropna(subset=["gid"])
    progress(0.88, "Building the event log")
    events, extra = build_events(det, identities, series, meta, settings, links)
    from .audio import audio_events, audio_series

    aud_path = out / "audio.csv"
    if not aud_path.exists():
        audio_series(Path(meta["path"])).to_csv(aud_path, index=False)
    aud = pd.read_csv(aud_path)
    snd = audio_events(aud)
    if snd:
        events = pd.concat([events.drop(columns=["id", "when"]), pd.DataFrame(snd)], ignore_index=True)
        events = events.sort_values(["start", "type"]).reset_index(drop=True)
        events.insert(0, "id", [f"E{i + 1:03d}" for i in range(len(events))])
        events["when"] = events.apply(lambda r: fmt_t(r.start) if r.duration == 0 else f"{fmt_t(r.start)}–{fmt_t(r.end)}",
                                      axis=1)
    extra["audio"] = {"has_audio": not aud.empty, "sound_events": len(snd)}
    if settings.captions_every > 0 and llm is not None and llm.configured:
        from .captions import caption_keyframes

        progress(0.9, "Describing keyframes with a vision-language model")
        cap = caption_keyframes(out, llm)
        if len(cap):
            events = pd.concat([events.drop(columns=["id"]), cap], ignore_index=True).sort_values("start")
            events.insert(0, "id", [f"E{i + 1:03d}" for i in range(len(events))])
    if len(identities):
        visible = {g: sum(e - s for s, e in segs) for g, segs in extra["segments"].items()}
        identities["visible_seconds"] = identities.id.map(visible).round(1)
        identities["first"] = identities.first_seen.map(fmt_t)
        identities["last"] = identities.last_seen.map(fmt_t)
    det.to_csv(out / "tracks_global.csv", index=False)
    identities.to_csv(out / "identities.csv", index=False)
    links.to_csv(out / "reid_links.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    (out / "segments.json").write_text(json.dumps(extra, default=float))
    a = Analysis(out, meta, settings, det, identities, links, events, series, extra["segments"])
    render_key = hashlib.sha256((out / "events.csv").read_bytes() + (out / "tracks_global.csv").read_bytes()).hexdigest()
    stamp = out / "annotated.key"
    if not a.annotated.exists() or not stamp.exists() or stamp.read_text() != render_key:
        progress(0.92, "Rendering annotated video")
        render_annotated(a)
        stamp.write_text(render_key)
    progress(1.0, f"Done: {len(identities)} identities, {len(events)} events")
    return a


# ----------------------------------------------------------------------------- rendering
def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def render_annotated(a: Analysis) -> Path:
    """Draw identities, zones and live events onto the video; encode H.264 so browsers can play it."""
    meta, settings = a.meta, a.settings
    W, H, fps = meta["width"], meta["height"], meta["fps"]
    scale = min(1.0, settings.render_width / W)
    w, h = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    det = a.detections
    by_frame = {f: g for f, g in det.groupby("frame")}
    sampled = np.array(sorted(by_frame))
    colours = {gid: PALETTE[i % len(PALETTE)] for i, gid in enumerate(sorted(det.gid.unique()))}
    ev = a.events
    banners = ev[ev.type.isin(["zone_enter", "zone_exit", "reappear", "activity_stop", "sudden_change", "interaction",
                               "sound"])]
    skip = 1 if meta["duration"] <= 180 else 2 if meta["duration"] <= 900 else 4  # long videos: lighter output
    proc = subprocess.Popen([_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", f"{fps / skip:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset",
                             "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(a.annotated)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(a.video))
    idx = 0
    while True:
        if idx % skip:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        frame = cv2.resize(frame, (w, h))
        for z in settings.zones:
            pts = np.array([[x * w, y * h] for x, y in z.points], np.int32)
            col = (0, 0, 255) if z.kind == "area" else (0, 200, 255)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], col)
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [pts], True, col, 2)
            cv2.putText(frame, z.name, tuple(pts[0] + [4, 18]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if len(sampled):
            k = int(np.searchsorted(sampled, idx, side="right") - 1)
            if k >= 0 and idx - sampled[k] <= meta["stride"]:
                for r in by_frame[sampled[k]].itertuples():
                    c = colours.get(r.gid, (255, 255, 255))
                    p1, p2 = (int(r.x1 * scale), int(r.y1 * scale)), (int(r.x2 * scale), int(r.y2 * scale))
                    cv2.rectangle(frame, p1, p2, c, 2)
                    label = f"{r.gid} {r.cls}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 6, p1[1]), c, -1)
                    cv2.putText(frame, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                                cv2.LINE_AA)
        live = banners[(banners.start <= t) & (t <= banners.start + 2.0)].tail(3)
        y = h - 12
        for r in live.iloc[::-1].itertuples():
            txt = f"{fmt_t(r.start)}  {r.details}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (8, y - th - 8), (16 + tw, y + 4), (20, 20, 20), -1)
            cv2.putText(frame, txt, (12, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y -= th + 14
        stamp = fmt_t(t)
        cv2.rectangle(frame, (w - 92, 6), (w - 6, 30), (20, 20, 20), -1)
        cv2.putText(frame, stamp, (w - 86, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(frame.tobytes())
        idx += 1
    cap.release()
    proc.stdin.close()
    proc.wait()
    return a.annotated


def frame_at(video: Path, t: float, width: int = 480, boxes: pd.DataFrame | None = None, meta: dict | None = None):
    """RGB still at time t (optionally with boxes for the given detections) — used as answer evidence."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if boxes is not None and len(boxes):
        for i, r in enumerate(boxes.itertuples()):
            c = PALETTE[i % len(PALETTE)]
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 3)
            cv2.putText(frame, str(r.gid), (int(r.x1) + 4, int(r.y1) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (width, int(h * width / w)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
