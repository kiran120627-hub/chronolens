"""Re-identification: link tracker tracklets into persistent identities.

The tracker loses an ID whenever someone is occluded for a while or leaves the frame. We link a
tracklet B to an earlier tracklet A when:
  * they never co-exist in time (one identity cannot be in two places),
  * they are the same kind of thing (person / vehicle / same object class),
  * they look alike  — CNN embedding cosine + clothing colour histograms (upper/lower body), and
  * the hand-over is physically plausible — distance between A's exit and B's entry vs. the gap
    duration, with a bonus for leaving/returning at a frame edge or re-appearing where they vanished.
Linking is greedy by score, each tracklet gets at most one predecessor and one successor.
Every link is reported with its score so the UI and the answers can show how sure we are.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import VEHICLES, Settings


def kind_of(cls: str) -> str:
    return "person" if cls == "person" else ("vehicle" if cls in VEHICLES else cls)


PREFIX = {"person": "P", "vehicle": "V", "backpack": "B", "handbag": "B", "suitcase": "B"}  # others -> "O"


@dataclass
class Tracklet:
    track: int
    cls: str
    kind: str
    start: float
    end: float
    n: int
    first_box: tuple
    last_box: tuple
    conf: float


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _near_edge(b, W, H, margin=0.04):
    return b[0] < W * margin or b[1] < H * margin or b[2] > W * (1 - margin) or b[3] > H * (1 - margin)


def build_tracklets(det: pd.DataFrame, settings: Settings) -> tuple[dict[int, Tracklet], pd.DataFrame]:
    out = {}
    keep = []
    for tid, g in det.groupby("track"):
        g = g.sort_values("t")
        dur = g.t.iloc[-1] - g.t.iloc[0]
        if dur < settings.min_track_seconds and len(g) < 4:
            continue  # flicker / false positive
        if dur < 3.0 and g.conf.mean() < 0.40:
            continue  # short AND weak: reflections, posters, faint shapes inside vehicles
        cls = g.cls.mode().iloc[0]
        f, l = g.iloc[0], g.iloc[-1]
        out[int(tid)] = Tracklet(int(tid), cls, kind_of(cls), float(f.t), float(l.t), len(g),
                                 (f.x1, f.y1, f.x2, f.y2), (l.x1, l.y1, l.x2, l.y2), float(g.conf.mean()))
        keep.append(tid)
    return out, det[det.track.isin(keep)].copy()


def appearance_similarity(a: int, b: int, app: dict) -> tuple[float, dict]:
    parts = {}
    ca, cb = app.get(f"{a}__cnn"), app.get(f"{b}__cnn")
    if ca is not None and cb is not None:
        cos = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9))
        parts["cnn"] = float(np.clip((cos - 0.80) / 0.18, 0, 1))  # ImageNet features: people are all ~0.8+ alike
    ha, hb = app.get(f"{a}__hsv"), app.get(f"{b}__hsv")
    if ha is not None and hb is not None:
        parts["colour"] = float(np.minimum(ha, hb).sum() / 2.0)  # histogram intersection, 0..1
    if not parts:
        return 0.0, parts
    if "cnn" in parts and "colour" in parts:
        return 0.35 * parts["cnn"] + 0.65 * parts["colour"], parts
    return next(iter(parts.values())), parts


def link_tracklets(tracklets: dict[int, Tracklet], app: dict, W: int, H: int, settings: Settings):
    diag = float(np.hypot(W, H))
    cands = []
    items = sorted(tracklets.values(), key=lambda x: x.start)
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a.kind != b.kind or b.start <= a.start:
                continue
            gap = b.start - a.end
            ca, cb = _center(a.last_box), _center(b.first_box)
            dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) / diag
            # a short overlap is allowed only for an occlusion split (same person seen on both sides of an occluder)
            if gap < -0.15 and not (gap >= -1.0 and dist < 0.12):
                continue
            if gap > settings.reid_max_gap:
                continue
            app_s, parts = appearance_similarity(a.track, b.track, app)
            allowed = 0.06 + 0.35 * max(gap, 0.0)  # fraction of the diagonal reachable in `gap` seconds
            penalty = 0.0 if dist <= allowed else min(0.5, (dist - allowed) * 1.5)
            bonus = 0.0
            a_edge, b_edge = _near_edge(a.last_box, W, H), _near_edge(b.first_box, W, H)
            if a_edge and b_edge:
                bonus += 0.05  # left the frame and came back
            elif not a_edge and dist < 0.08:
                bonus += 0.04  # vanished mid-frame (occlusion) and re-appeared at the same place
            score = app_s + bonus - penalty - 0.004 * max(gap, 0.0)  # prefer the most recent plausible hand-over
            reason = (f"appearance {app_s:.2f} (" + ", ".join(f"{k} {v:.2f}" for k, v in parts.items()) + f"), "
                      f"gap {gap:.1f}s, hand-over distance {dist:.2f}·diag"
                      + (", edge exit→edge return" if a_edge and b_edge else "")
                      + (", re-appeared where lost" if (not a_edge and dist < 0.08) else "")
                      + (f", implausible jump −{penalty:.2f}" if penalty else ""))
            cands.append((score, a.track, b.track, gap, reason))
    # online assignment: in order of appearance, each new tracklet takes its best still-available predecessor
    succ, pred, links = {}, {}, []
    by_b: dict[int, list] = {}
    for c in cands:
        by_b.setdefault(c[2], []).append(c)
    for b in [t.track for t in items]:
        for score, a, _, gap, reason in sorted(by_b.get(b, []), reverse=True):
            if score < settings.reid_threshold:
                break
            if a in succ:
                continue
            succ[a], pred[b] = b, a
            links.append({"from_track": a, "to_track": b, "score": round(score, 3), "gap": round(gap, 2), "reason": reason})
            break
    # chains -> identities
    gid_of: dict[int, str] = {}
    counters: dict[str, int] = {}
    identities = []
    for t in sorted(tracklets.values(), key=lambda x: x.start):
        if t.track in pred:
            continue
        chain = [t.track]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        kind = t.kind
        prefix = PREFIX.get(kind, "O")
        counters[prefix] = counters.get(prefix, 0) + 1
        gid = f"{prefix}{counters[prefix]}"
        for tr in chain:
            gid_of[tr] = gid
        members = [tracklets[c] for c in chain]
        chain_links = [lk for lk in links if lk["from_track"] in chain]
        identities.append({
            "id": gid, "kind": kind, "cls": pd.Series([m.cls for m in members]).mode().iloc[0],
            "first_seen": min(m.start for m in members), "last_seen": max(m.end for m in members),
            "tracklets": chain, "reid_links": len(chain_links),
            "reid_score": round(min([lk["score"] for lk in chain_links], default=1.0), 3),
            "thumb_track": max(members, key=lambda m: m.n * m.conf).track,
        })
    for lk in links:
        lk["identity"] = gid_of.get(lk["from_track"])
    return gid_of, pd.DataFrame(identities), pd.DataFrame(links)
