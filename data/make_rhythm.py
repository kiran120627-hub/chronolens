"""Render a machine-rhythm benchmark video with exact ground truth.

A machine (spinning fan) normally stops every 30 s for 4 s. Two stops break the rhythm:
  * one comes 9 s early
  * one lasts 10 s instead of 4 s
Outputs data/videos/machine_rhythm.mp4 and eval/questions_rhythm.json.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_VIDEO = ROOT / "data" / "videos" / "machine_rhythm.mp4"
OUT_Q = ROOT / "eval" / "questions_rhythm.json"
W, H, FPS, DUR = 960, 540, 25, 215.0
MACHINE = (0.35, 0.18, 0.65, 0.72)
#          start, duration    (normal: every 30 s for 4 s)
STOPS = [(15, 4), (45, 4), (75, 4), (96, 4), (135, 4), (165, 10), (195, 4)]
UNEXPECTED = [(96, "came 9 s early"), (165, "lasted 10 s instead of 4 s")]


def main() -> None:
    import imageio_ffmpeg

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    bg = np.full((H, W, 3), (62, 66, 70), np.uint8)
    cv2.rectangle(bg, (0, int(H * 0.8)), (W, H), (88, 92, 96), -1)
    x1, y1, x2, y2 = (int(MACHINE[0] * W), int(MACHINE[1] * H), int(MACHINE[2] * W), int(MACHINE[3] * H))
    cv2.rectangle(bg, (x1, y1), (x2, y2), (110, 128, 136), -1)
    cv2.rectangle(bg, (x1, y1), (x2, y2), (40, 40, 40), 3)
    cv2.putText(bg, "PRESS 04", (x1 + 12, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                           "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT_VIDEO)],
                          stdin=subprocess.PIPE)
    angle = 0.0
    cx, cy, r = (x1 + x2) // 2, (y1 + y2) // 2 + 10, int((x2 - x1) * 0.33)
    for i in range(int(DUR * FPS)):
        t = i / FPS
        running = not any(s <= t < s + d for s, d in STOPS)
        if running:
            angle += 0.42
        f = bg.copy()
        cv2.circle(f, (cx, cy), r + 8, (30, 30, 30), 4)
        for k in range(4):
            a = angle + k * math.pi / 2
            cv2.line(f, (cx, cy), (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))), (225, 225, 225), 12)
        cv2.circle(f, (x1 + 22, y2 - 22), 10, (60, 210, 60) if running else (60, 60, 210), -1)
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()
    zone = [(MACHINE[0], MACHINE[1]), (MACHINE[2], MACHINE[1]), (MACHINE[2], MACHINE[3]), (MACHINE[0], MACHINE[3])]
    meta = {"video": str(OUT_VIDEO.relative_to(ROOT)).replace("\\", "/"),
            "zones": [{"name": "Machine", "kind": "activity", "points": zone}],
            "stops": STOPS, "unexpected": UNEXPECTED, "time_tolerance": 1.5,
            "questions": [
                {"id": "R1", "question": "How many times did the machine stop?",
                 "expected": {"value": len(STOPS), "times": [s for s, _ in STOPS]}},
                {"id": "R2", "question": "How many times did the machine stop unexpectedly?",
                 "expected": {"value": len(UNEXPECTED), "times": [s for s, _ in UNEXPECTED]}},
                {"id": "R3", "question": "What is the machine's normal cycle?",
                 "expected": {"value": 30.0, "value_tol": 3.0}},
            ]}
    OUT_Q.write_text(json.dumps(meta, indent=2))
    print("wrote", OUT_VIDEO)


if __name__ == "__main__":
    main()
