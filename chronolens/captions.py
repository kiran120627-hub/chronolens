"""Optional: describe keyframes with a vision-language model and add them to the event log as `caption` events.

Covers things a COCO detector cannot name ("worker opens the panel", "alarm light is on").
Captions carry their exact keyframe timestamp; they are evidence, never the source of a time.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from .config import fmt_t
from .llm import LLM, extract_json

SYSTEM = """You describe surveillance / workplace video keyframes for a temporal-reasoning system.
For each frame (given with its timestamp) write one short factual caption of visible actions and states
(people's actions, vehicles, machines running or stopped, lights/alarms, objects being carried or left).
Do not guess identities or intentions. Reply with a ```json list: [{"t": <seconds>, "caption": "..."}]"""


def caption_keyframes(out_dir: Path, llm: LLM, batch: int = 8) -> pd.DataFrame:
    frames = sorted((out_dir / "keyframes").glob("*.jpg"))
    rows = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        parts = [{"type": "text", "text": "Frames in chronological order:"}]
        for f in chunk:
            t = float(f.stem)
            parts.append({"type": "text", "text": f"t={t:.2f}s ({fmt_t(t)})"})
            parts.append({"type": "image", "b64": base64.b64encode(f.read_bytes()).decode(), "mime": "image/jpeg"})
        try:
            items = extract_json(llm.complete(SYSTEM, [{"role": "user", "content": parts}], tag="captions"))
        except Exception as e:  # noqa: BLE001
            print(f"[chronolens] captioning failed for batch {i // batch}: {e}")
            continue
        for it in items if isinstance(items, list) else []:
            try:
                t = float(it["t"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({"type": "caption", "start": round(t, 2), "end": round(t, 2), "duration": 0.0, "subject": "",
                         "kind": "", "cls": "", "zone": "", "other": "", "details": str(it.get("caption", ""))[:240],
                         "confidence": 0.6, "when": fmt_t(t)})
    return pd.DataFrame(rows)
