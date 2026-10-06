"""Event engine: turn tracks + motion signals into a timestamped event log (deterministic, no AI).

Event types
  enter_view / exit_view      identity first appears / finally leaves (or video ends)
  occluded / reappear         identity lost mid-video and found again (re-identification)
  zone_enter / zone_exit      anchor point (feet for people) crosses an area-zone boundary (with hysteresis)
  in_zone                     interval an identity spends inside a zone
  stationary                  identity does not move for >= stationary_min seconds
  untouched                   non-person object stationary with no person interacting
  interaction                 person close to / touching an object or vehicle
  activity_stop / activity_start   motion inside an activity zone (e.g. a machine) stops / resumes
  sudden_change               spike in global motion, brightness or red light (alarms, flashes, commotion)
  caption                     optional keyframe description from a vision-language model
Each event: id, type, start, end, duration, subject, kind, cls, zone, other, details, confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Settings, fmt_t, point_in_poly

COLUMNS = ["id", "type", "start", "end", "duration", "subject", "kind", "cls", "zone", "other", "details", "confidence"]


def _runs(times: np.ndarray, flags: np.ndarray):
    """Contiguous runs of equal flag values -> list of (flag, start_t, end_t, i0, i1)."""
    out = []
    if len(times) == 0:
        return out
    i0 = 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[i0]:
            out.append((bool(flags[i0]), float(times[i0]), float(times[i - 1]), i0, i - 1))
            i0 = i
    return out


def _segments(ts: np.ndarray, gap: float):
    if len(ts) == 0:
        return []
    cuts = np.where(np.diff(ts) > gap)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(ts) - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _merge_intervals(iv, max_gap):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _near_edge(row, W, H, m=0.04):
    return row.x1 < W * m or row.y1 < H * m or row.x2 > W * (1 - m) or row.y2 > H * (1 - m)


def build_events(det: pd.DataFrame, identities: pd.DataFrame, series: pd.DataFrame, meta: dict,
                 settings: Settings, links: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    W, H, dur = meta["width"], meta["height"], meta["duration"]
    step = 1.0 / meta.get("sample_fps", 10.0)
    ev: list[dict] = []

    def add(type_, start, end=None, subject="", kind="", cls="", zone="", other="", details="", conf=1.0):
        end = start if end is None else end
        ev.append({"type": type_, "start": round(float(start), 2), "end": round(float(end), 2),
                   "duration": round(float(end) - float(start), 2), "subject": subject, "kind": kind, "cls": cls,
                   "zone": zone, "other": other, "details": details, "confidence": round(float(conf), 2)})

    segments: dict[str, list[tuple[float, float]]] = {}
    id_info = identities.set_index("id").to_dict("index") if len(identities) else {}
    area_zones = [z for z in settings.zones if z.kind == "area"]

    for gid, g in det.groupby("gid"):
        g = g.sort_values("t").reset_index(drop=True)
        info = id_info.get(gid, {})
        kind, cls = info.get("kind", ""), info.get("cls", g.cls.iloc[0])
        ts = g.t.to_numpy()
        segs = _segments(ts, settings.segment_gap)
        segments[gid] = [(float(ts[s]), float(ts[e])) for s, e in segs]
        conf_all = float(g.conf.mean())

        # ---- visibility
        for k, (s, e) in enumerate(segs):
            r0, r1 = g.iloc[s], g.iloc[e]
            if k == 0:
                add("enter_view", ts[s], subject=gid, kind=kind, cls=cls, conf=r0.conf,
                    details="enters from frame edge" if _near_edge(r0, W, H) else "first seen inside the frame")
            else:
                gap = ts[s] - ts[segs[k - 1][1]]
                add("reappear", ts[s], subject=gid, kind=kind, cls=cls, conf=min(r0.conf, info.get("reid_score", 1.0)),
                    details=f"seen again after {gap:.1f}s unseen (re-identified, link score {info.get('reid_score', 1.0):.2f})")
            if k < len(segs) - 1:
                add("occluded" if not _near_edge(r1, W, H) else "exit_view", ts[e], subject=gid, kind=kind, cls=cls,
                    conf=r1.conf, details=("lost mid-frame (likely hidden behind something)" if not _near_edge(r1, W, H)
                                           else "leaves the frame (comes back later)"))
            elif ts[e] < dur - 2 * step - 0.5:
                add("exit_view", ts[e], subject=gid, kind=kind, cls=cls, conf=r1.conf,
                    details="leaves the frame" if _near_edge(r1, W, H) else "disappears mid-frame")

        # ---- zones
        if area_zones:
            if kind == "person":
                ax, ay = (g.x1 + g.x2) / 2 / W, g.y2 / H  # feet
            else:
                ax, ay = (g.x1 + g.x2) / 2 / W, (g.y1 + g.y2) / 2 / H
            for z in area_zones:
                inside = np.array([point_in_poly(x, y, z.points) for x, y in zip(ax, ay)])
                presence = []
                for s, e in segs:
                    seg_t = ts[s:e + 1]
                    flags = inside[s:e + 1].copy()
                    # hysteresis: drop runs shorter than zone_min_dwell
                    for f, rs, re_, i0, i1 in _runs(seg_t, flags):
                        if re_ - rs + step < settings.zone_min_dwell and 0 < i0 and i1 < len(flags) - 1:
                            flags[i0:i1 + 1] = not f
                    for f, rs, re_, _, _ in _runs(seg_t, flags):
                        if f:
                            presence.append((rs, re_))
                for s, e in _merge_intervals(presence, 3.0):
                    add("zone_enter", s, subject=gid, kind=kind, cls=cls, zone=z.name, conf=conf_all,
                        details=f"{gid} enters {z.name}")
                    add("in_zone", s, e, subject=gid, kind=kind, cls=cls, zone=z.name, conf=conf_all,
                        details=f"{gid} inside {z.name} for {e - s:.1f}s")
                    if True:  # exits at end of video are flagged in the details
                        add("zone_exit", e, subject=gid, kind=kind, cls=cls, zone=z.name, conf=conf_all,
                            details=f"{gid} leaves {z.name}" + (" (end of video)" if e >= dur - 1 else ""))

        # ---- stationary
        cx, cy = ((g.x1 + g.x2) / 2).to_numpy(), ((g.y1 + g.y2) / 2).to_numpy()
        hh = (g.y2 - g.y1).to_numpy()
        for s, e in segs:
            i = s
            while i <= e:
                j = i
                tol = settings.stationary_tol * max(np.median(hh[i:min(e, i + 10) + 1]), 8.0)
                while j + 1 <= e and np.hypot(cx[j + 1] - cx[i], cy[j + 1] - cy[i]) <= tol:
                    j += 1
                if ts[j] - ts[i] >= settings.stationary_min:
                    add("stationary", ts[i], ts[j], subject=gid, kind=kind, cls=cls, conf=conf_all,
                        details=f"{gid} does not move for {ts[j] - ts[i]:.1f}s")
                i = j + 1

    # ---- interactions (person <-> object/vehicle) and untouched objects
    people = det[det.gid.map(lambda x: str(x).startswith("P"))]
    others = det[~det.gid.map(lambda x: str(x).startswith("P"))]
    contact: dict[str, list] = {}
    if len(people) and len(others):
        for oid, og in others.groupby("gid"):
            m = og.merge(people, on="t", suffixes=("", "_p"))
            if m.empty:
                continue
            ow, oh = m.x2 - m.x1, m.y2 - m.y1
            mx, my = ow * settings.interaction_margin, oh * settings.interaction_margin
            # overlap between the person box and the (slightly expanded) object box, relative to the object's area
            iw = (np.minimum(m.x2_p, m.x2 + mx) - np.maximum(m.x1_p, m.x1 - mx)).clip(lower=0)
            ih = (np.minimum(m.y2_p, m.y2 + my) - np.maximum(m.y1_p, m.y1 - my)).clip(lower=0)
            touch = (iw * ih) / (ow * oh + 1e-6) >= 0.15
            for pid, pm in m[touch].groupby("gid_p"):
                tt = np.sort(pm.t.unique())
                for s, e in _segments(tt, 1.0):
                    if tt[e] - tt[s] >= 0.8:  # passing by is not an interaction
                        contact.setdefault(oid, []).append((float(tt[s]), float(tt[e])))
                        okind = id_info.get(oid, {}).get("kind", "")
                        add("interaction", tt[s], tt[e], subject=pid, kind="person", cls="person", other=oid,
                            conf=float(pm.conf.mean()),
                            details=f"{pid} {'next to' if okind == 'vehicle' else 'touches / is next to'} {oid} "
                                    f"({id_info.get(oid, {}).get('cls', '')}) for {tt[e] - tt[s]:.1f}s")
    for e in [x for x in ev if x["type"] == "stationary" and not str(x["subject"]).startswith("P")]:
        free = [(e["start"], e["end"])]
        for cs, ce in contact.get(e["subject"], []):
            nxt = []
            for fs, fe in free:
                if ce < fs or cs > fe:
                    nxt.append((fs, fe))
                else:
                    if cs > fs:
                        nxt.append((fs, cs))
                    if ce < fe:
                        nxt.append((ce, fe))
            free = nxt
        for fs, fe in free:
            if fe - fs >= settings.stationary_min:
                ev.append({**e, "type": "untouched", "start": fs, "end": fe, "duration": round(fe - fs, 2),
                           "details": f"{e['subject']} ({e['cls']}) untouched for {fe - fs:.1f}s"})

    # ---- activity zones (machines)
    activity_summary = {}
    for z in [z for z in settings.zones if z.kind == "activity"]:
        col = f"act::{z.name}"
        if col not in series:
            continue
        t = series.t.to_numpy()
        e_ = series[col].to_numpy()
        lo, hi = np.percentile(e_[1:], 5), np.percentile(e_[1:], 95)
        activity_summary[z.name] = {"low": float(lo), "high": float(hi)}
        if hi - lo < 0.6:
            continue  # never clearly active
        thr = lo + 0.35 * (hi - lo)
        flags = e_ > thr
        flags[0] = flags[1] if len(flags) > 1 else flags[0]
        runs = _runs(t, flags)
        for f, rs, re_, i0, i1 in runs:  # fill short gaps
            if not f and re_ - rs + step < settings.activity_min_stop and 0 < i0 and i1 < len(flags) - 1:
                flags[i0:i1 + 1] = True
        for f, rs, re_, i0, i1 in _runs(t, flags):
            if f and re_ - rs + step < 0.5 and 0 < i0 and i1 < len(flags) - 1:
                flags[i0:i1 + 1] = False
        runs = _runs(t, flags)
        contrast = float(np.clip((hi - lo) / (hi + 1e-6), 0, 1))
        for k, (f, rs, re_, i0, i1) in enumerate(runs):
            if not f and k > 0:
                resumed = k < len(runs) - 1
                end = runs[k + 1][1] if resumed else re_
                add("activity_stop", rs, end, zone=z.name, conf=0.6 + 0.4 * contrast,
                    details=f"motion in {z.name} stops for {end - rs:.1f}s" + ("" if resumed else " (until end of video)"))
                if resumed:
                    add("activity_start", end, zone=z.name, conf=0.6 + 0.4 * contrast,
                        details=f"motion in {z.name} resumes")

    # ---- sudden changes
    if len(series) > 5:
        t = series.t.to_numpy()
        for col, label, sign in (("motion", "sudden burst of motion", 1), ("brightness", "sudden brightness change", 0),
                                 ("redness", "red light / flashing", 1)):
            x = series[col].to_numpy().astype(float)
            d = np.abs(np.diff(x, prepend=x[0])) if sign == 0 else x - np.median(x)
            mad = np.median(np.abs(d - np.median(d))) + 1e-6
            z = 0.6745 * (d - np.median(d)) / mad
            thr = 8.0 if col != "redness" else 6.0
            flags = (z > thr) & (np.abs(d) > (1.5 if col != "motion" else 2.0))
            for f, rs, re_, _, _ in _runs(t, flags):
                if f:
                    add("sudden_change", rs, re_, zone="whole frame", conf=float(min(1.0, 0.5 + z[(t >= rs) & (t <= re_)].max() / 40)),
                        details=label)
        # merge adjacent sudden_change events of the same label
        sc = sorted([e for e in ev if e["type"] == "sudden_change"], key=lambda e: (e["details"], e["start"]))
        merged = []
        for e in sc:
            if merged and merged[-1]["details"] == e["details"] and e["start"] - merged[-1]["end"] <= 1.5:
                merged[-1]["end"] = e["end"]
                merged[-1]["duration"] = round(merged[-1]["end"] - merged[-1]["start"], 2)
            else:
                merged.append(dict(e))
        ev = [e for e in ev if e["type"] != "sudden_change"] + merged

    df = pd.DataFrame(ev, columns=[c for c in COLUMNS if c != "id"])
    order = {"enter_view": 0, "reappear": 1, "zone_enter": 2, "in_zone": 3, "interaction": 4, "stationary": 5,
             "untouched": 6, "activity_stop": 7, "activity_start": 8, "sudden_change": 9, "zone_exit": 10,
             "occluded": 11, "exit_view": 12, "caption": 13}
    df["_o"] = df["type"].map(order).fillna(99)
    df = df.sort_values(["start", "_o", "subject"]).drop(columns="_o").reset_index(drop=True)
    df.insert(0, "id", [f"E{i + 1:03d}" for i in range(len(df))])
    df["when"] = df.apply(lambda r: fmt_t(r.start) if r.duration == 0 else f"{fmt_t(r.start)}–{fmt_t(r.end)}", axis=1)
    return df, {"segments": segments, "activity": activity_summary}
