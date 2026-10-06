"""Dev diagnostic: re-run post-processing on the cached benchmark analysis and print identities/links/similarities."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 100); pd.set_option("display.max_rows", 300)
from chronolens.config import Settings, Zone
from chronolens.pipeline import analyze
from chronolens.reid import appearance_similarity
m = json.load(open("eval/questions_sim.json"))
s = Settings(zones=[Zone(z["name"], [tuple(p) for p in z["points"]], z["kind"]) for z in m["zones"]])
a = analyze(m["video"], s)
print(a.identities[["id", "cls", "first_seen", "last_seen", "tracklets", "reid_score", "visible_seconds"]].to_string())
print(a.links.to_string())
if "--sim" in sys.argv:
    app = dict(np.load(a.dir / "appearance.npz"))
    tr = sorted({int(k.split("__")[0]) for k in app})
    print("track ids", tr)
    for x in tr:
        print(x, " ".join(f"{appearance_similarity(x, y, app)[0]:.2f}" for y in tr))
if "--events" in sys.argv:
    print(a.events[a.events.type != "in_zone"][["id", "type", "when", "subject", "zone", "other", "details"]].to_string(index=False))
