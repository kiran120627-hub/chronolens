"""Unit tests for re-identification + event engine on hand-made tracks (no GPU / video needed).

Run:  python tests/test_engine.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chronolens.config import Settings, Zone  # noqa: E402
from chronolens.events import build_events  # noqa: E402
from chronolens.reid import build_tracklets, link_tracklets  # noqa: E402

W, H, FPS = 1000, 500, 10
fails = []


def check(c, msg):
    print(("  PASS  " if c else "  FAIL  ") + msg)
    if not c:
        fails.append(msg)


def walk(track, cls, t0, t1, x0, x1, y_feet, h=120, w=50):
    rows = []
    for t in np.arange(t0, t1 + 1e-9, 1 / FPS):
        u = (t - t0) / max(t1 - t0, 1e-9)
        x = x0 + (x1 - x0) * u
        rows.append({"t": round(t, 2), "frame": int(round(t * 25)), "track": track, "cls": cls, "conf": 0.9,
                     "x1": x - w / 2, "y1": y_feet - h, "x2": x + w / 2, "y2": y_feet})
    return rows


# person A: walks right through the zone (x>600) 0-10s, leaves; returns 20-30s (new tracker id 5)
# person B: different look, appears 12s, stands still 13-30s
# a bag (suitcase) sits still 2-40s, person B touches it at 25-27s
rows = walk(1, "person", 0, 10, -20, 1020, 450) + walk(5, "person", 20, 30, 1020, 300, 450)
rows += walk(2, "person", 12, 13, 100, 200, 300) + walk(2, "person", 13.1, 30, 200, 200, 300)
rows += walk(9, "suitcase", 2, 40, 260, 260, 330, h=40, w=40)
det = pd.DataFrame(rows)
# B's box overlaps the bag only 25-27 s: move B next to the bag then
det.loc[(det.track == 2) & (det.t >= 25) & (det.t <= 27), ["x1", "x2"]] = [235.0, 285.0]
app = {"1__hsv": np.r_[np.ones(128), np.zeros(128)] / 128, "5__hsv": np.r_[np.ones(128), np.zeros(128)] / 128,
       "2__hsv": np.r_[np.zeros(128), np.ones(128)] / 128, "9__hsv": np.ones(256) / 256}
app = {k: v.astype(np.float32) * 2 for k, v in app.items()}  # two halves sum to 1 each

s = Settings(stationary_min=8, zones=[Zone.rect("Restricted", 0.6, 0.5, 1.0, 1.0)])
tr, det2 = build_tracklets(det, s)
gid_of, ids, links = link_tracklets(tr, app, W, H, s)
print("identities:", ids[["id", "tracklets", "reid_score"]].to_dict("records"))
check(gid_of[1] == gid_of[5], "person A re-identified after leaving and returning (tracks 1 & 5)")
check(gid_of[2] != gid_of[1], "person B kept separate")
det2["gid"] = det2.track.map(gid_of)
meta = {"width": W, "height": H, "duration": 40.0, "sample_fps": FPS}
series = pd.DataFrame({"t": np.arange(0, 40, 0.1), "motion": 1.0, "brightness": 100.0, "redness": 0.0})
ev, extra = build_events(det2, ids, series, meta, s)
print(ev[["id", "type", "when", "subject", "zone", "other", "details"]].to_string(index=False))
A = gid_of[1]
ze = ev[(ev.type == "zone_enter") & (ev.subject == A)]
check(len(ze) == 2, f"A enters the zone twice (got {len(ze)})")
check(abs(ze.start.iloc[0] - 6.05) < 0.25, f"first zone entry at ~6.0s (got {ze.start.iloc[0]})")
check(((ev.type == "reappear") & (ev.subject == A)).sum() == 1, "A has a reappear event")
st_ = ev[(ev.type == "stationary") & (ev.subject == gid_of[2])]
check(len(st_) >= 1 and st_.duration.max() > 10, "B stationary for >10s")
it = ev[ev.type == "interaction"]
check(len(it) == 1 and abs(it.start.iloc[0] - 25) < 0.3, "B interacts with the bag at ~25s")
un = ev[ev.type == "untouched"]
check(len(un) == 2, f"bag untouched before and after the interaction (got {len(un)})")
print("\nRESULT:", "ALL PASSED" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
