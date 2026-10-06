"""Render a benchmark video with exactly known ground truth.

Real people and a real vehicle are cut out (YOLO11 segmentation) of the sample photos that ship with
ultralytics, recoloured into distinct identities and animated over a synthetic workshop scene that contains:
  * a restricted floor area (red hatched)            -> zone entries/exits
  * a pillar in the foreground                       -> occlusion (identity must survive it)
  * people leaving the frame and coming back         -> re-identification
  * a machine with a spinning fan that stops twice   -> activity stops
  * an alarm lamp that flashes red                   -> "what happened right before the alarm?"
  * a delivery vehicle that arrives, parks and leaves
  * a person who stands still for a long time        -> loitering / stationary
Outputs data/videos/warehouse_sim.mp4 and eval/questions_sim.json (with ground-truth times).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_VIDEO = ROOT / "data" / "videos" / "warehouse_sim.mp4"
OUT_Q = ROOT / "eval" / "questions_sim.json"
W, H, FPS, DUR = 1280, 720, 25, 100.0
ZONE = [(0.62, 0.62), (0.97, 0.62), (0.97, 0.99), (0.62, 0.99)]  # restricted area (normalised)
MACHINE = (0.56, 0.05, 0.70, 0.28)
LAMP = (0.90, 0.05, 0.96, 0.15)
PILLAR = (0.44, 0.0, 0.49, 1.0)
MACHINE_STOPS = [(20.0, 27.0), (61.0, 66.0)]
ALARM = (70.0, 74.0)


def cutouts():
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    model = YOLO("yolo11n-seg.pt")
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    r = model(img, verbose=False)[0]
    people, vehicle = [], None
    for box, mask, c in zip(r.boxes.xyxy.cpu().numpy(), r.masks.data.cpu().numpy(), r.boxes.cls.int().cpu().numpy()):
        m = cv2.resize(mask, (img.shape[1], img.shape[0])) > 0.5
        x1, y1, x2, y2 = box.astype(int)
        rgba = np.dstack([img, (m * 255).astype(np.uint8)])[y1:y2, x1:x2]
        if c == 0 and (y2 - y1) > 300 and x1 > 3 and x2 < img.shape[1] - 3:  # complete people only
            people.append((x1, rgba))
        elif c == 5:
            vehicle = rgba
    people = [p for _, p in sorted(people, key=lambda p: p[0])]
    return people, vehicle


def recolour(sprite: np.ndarray, hue_shift: int, sat: float) -> np.ndarray:
    out = sprite.copy()
    bgr = out[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h = sprite.shape[0]
    body = slice(int(h * 0.22), h)  # leave the head alone
    hsv[body, :, 0] = (hsv[body, :, 0] + hue_shift) % 180
    hsv[body, :, 1] = np.clip(hsv[body, :, 1] * sat + 60, 0, 255)
    hsv[body, :, 2] = np.clip(hsv[body, :, 2] * 1.25 + 25, 0, 255)
    out[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def paste(frame, sprite, cx, feet_y, height):
    s = height / sprite.shape[0]
    sp = cv2.resize(sprite, (max(2, int(sprite.shape[1] * s)), max(2, int(height))))
    h, w = sp.shape[:2]
    x1, y1 = int(cx - w / 2), int(feet_y - h)
    fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(W, x1 + w), min(H, y1 + h)
    if fx1 >= fx2 or fy1 >= fy2:
        return None
    crop = sp[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    a = crop[:, :, 3:4] / 255.0
    frame[fy1:fy2, fx1:fx2] = (frame[fy1:fy2, fx1:fx2] * (1 - a) + crop[:, :, :3] * a).astype(np.uint8)
    return (fx1, fy1, fx2, fy2)


def lerp_path(t, keys):
    """keys: list of (time, x, y). Returns (x, y) or None when outside the key range."""
    if t < keys[0][0] or t > keys[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return None


# trajectories: (time, x_norm of feet, y_norm of feet). Off-screen x < -0.05 or > 1.05.
PEOPLE = {
    "P1": [(3.0, -0.06, 0.86), (18.0, 1.08, 0.86), (40.0, 1.08, 0.50), (52.0, -0.06, 0.50)],
    "P2": [(33.0, -0.06, 0.80), (41.5, 0.70, 0.80), (45.5, 0.70, 0.80), (49.0, 0.40, 0.70), (60.0, 0.40, 0.70),
           (67.0, 0.66, 0.78), (73.0, 0.80, 0.78), (76.0, 1.08, 0.78)],
    "P3": [(55.0, -0.06, 0.66), (58.0, 0.28, 0.66), (92.0, 0.28, 0.66), (97.0, -0.06, 0.66)],
}
VEHICLE = [(25.0, -0.30, 0.52), (30.0, 0.22, 0.52), (80.0, 0.22, 0.52), (86.0, -0.32, 0.52)]


def in_zone(x, y):
    return ZONE[0][0] <= x <= ZONE[1][0] and ZONE[0][1] <= y <= ZONE[2][1]


def visible_x(x):
    return -0.02 < x < 1.02


def ground_truth():
    """Exact event times from the trajectories (feet point in zone, visibility)."""
    ts = np.arange(0, DUR, 0.02)
    gt = {"zone_entries": [], "zone_exits": [], "first_seen": {}, "last_seen": {}}
    for pid, keys in PEOPLE.items():
        prev_in, prev_vis = False, False
        for t in ts:
            p = lerp_path(t, keys)
            vis = p is not None and visible_x(p[0])
            inside = vis and in_zone(*p)
            if vis and pid not in gt["first_seen"]:
                gt["first_seen"][pid] = round(t, 2)
            if vis:
                gt["last_seen"][pid] = round(t, 2)
            if inside and not prev_in:
                gt["zone_entries"].append((pid, round(t, 2)))
            if prev_in and not inside:
                gt["zone_exits"].append((pid, round(t, 2)))
            prev_in, prev_vis = inside, vis
    v_seen = [t for t in ts if (p := lerp_path(t, VEHICLE)) is not None and p[0] + 0.2 > 0]
    gt["vehicle_first_seen"] = round(v_seen[0], 2)
    gt["vehicle_parked"] = (30.0, 80.0)
    return gt


def main() -> None:
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    people, bus = cutouts()
    assert len(people) >= 2 and bus is not None, "could not extract sprites"
    sprites = {"P1": recolour(people[0], 0, 1.6),                 # warm / orange outfit
               "P2": recolour(people[1], 100, 1.8),               # blue outfit
               "P3": recolour(people[1][:, ::-1].copy(), 150, 1.8)}  # mirrored + magenta (similar build to P2!)

    # static background
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        c = 70 + int(60 * y / H) if y > H * 0.42 else 150 - int(40 * y / (H * 0.42))
        bg[y, :] = (c, c + 6, c + 10) if y > H * 0.42 else (c + 15, c + 10, c)
    cv2.rectangle(bg, (0, int(H * 0.42)), (W, int(H * 0.44)), (60, 60, 60), -1)
    zx1, zy1, zx2, zy2 = int(ZONE[0][0] * W), int(ZONE[0][1] * H), int(ZONE[1][0] * W), int(ZONE[2][1] * H)
    overlay = bg.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (40, 40, 200), -1)
    bg = cv2.addWeighted(overlay, 0.25, bg, 0.75, 0)
    hatch = bg.copy()
    for k in range(-H, W, 40):
        cv2.line(hatch, (zx1 + k, zy1), (zx1 + k + (zy2 - zy1), zy2), (40, 60, 230), 3)
    bg[zy1:zy2, zx1:zx2] = hatch[zy1:zy2, zx1:zx2]  # hatching only inside the zone
    cv2.rectangle(bg, (zx1, zy1), (zx2, zy2), (30, 30, 230), 4)
    cv2.putText(bg, "RESTRICTED", (zx1 + 20, zy1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 240), 3)
    mx1, my1, mx2, my2 = (int(MACHINE[0] * W), int(MACHINE[1] * H), int(MACHINE[2] * W), int(MACHINE[3] * H))
    cv2.rectangle(bg, (mx1, my1), (mx2, my2), (90, 110, 120), -1)
    cv2.rectangle(bg, (mx1, my1), (mx2, my2), (40, 40, 40), 3)
    lx1, ly1, lx2, ly2 = (int(LAMP[0] * W), int(LAMP[1] * H), int(LAMP[2] * W), int(LAMP[3] * H))
    cv2.rectangle(bg, (lx1, ly1), (lx2, ly2), (50, 50, 50), -1)

    ff = subprocess.Popen([_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                           f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT_VIDEO)], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    angle = 0.0
    for i in range(n):
        t = i / FPS
        f = bg.copy()
        # machine fan
        running = not any(s <= t < e for s, e in MACHINE_STOPS)
        if running:
            angle += 0.45
        cx, cy, r = (mx1 + mx2) // 2, (my1 + my2) // 2, int((mx2 - mx1) * 0.38)
        cv2.circle(f, (cx, cy), r + 6, (30, 30, 30), 3)
        for k in range(4):
            a = angle + k * math.pi / 2
            cv2.line(f, (cx, cy), (int(cx + r * math.cos(a)), int(cy + r * math.sin(a))), (220, 220, 220), 9)
        cv2.circle(f, (mx1 + 18, my2 - 16), 8, (60, 220, 60) if running else (60, 60, 200), -1)
        # alarm lamp (flashing 4 Hz) + red wash over the frame
        if ALARM[0] <= t < ALARM[1] and int(t * 4) % 2 == 0:
            cv2.rectangle(f, (lx1, ly1), (lx2, ly2), (0, 0, 255), -1)
            red = np.zeros_like(f)
            red[:, :, 2] = 255
            f = cv2.addWeighted(red, 0.18, f, 0.82, 0)
        # vehicle
        p = lerp_path(t, VEHICLE)
        if p:
            paste(f, bus, p[0] * W, p[1] * H, 300)
        # people (draw farther ones first)
        order = sorted(PEOPLE, key=lambda k: (lerp_path(t, PEOPLE[k]) or (0, 0))[1])
        for pid in order:
            p = lerp_path(t, PEOPLE[pid])
            if p is None:
                continue
            x, y = p
            height = 140 + 330 * (y - 0.45) / 0.5
            bob = 3 * math.sin(t * 9 + hash(pid) % 7)
            paste(f, sprites[pid], x * W, y * H + bob, height)
        # foreground pillar (occluder)
        cv2.rectangle(f, (int(PILLAR[0] * W), 0), (int(PILLAR[2] * W), H), (75, 85, 95), -1)
        cv2.rectangle(f, (int(PILLAR[0] * W), 0), (int(PILLAR[2] * W), H), (45, 50, 55), 3)
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    ff.wait()

    gt = ground_truth()
    entries = gt["zone_entries"]
    p2_entries = [t for pid, t in entries if pid == "P2"]
    p1_entries = [t for pid, t in entries if pid == "P1"]
    p2_time = sum(x - e for (pid, e), (pid2, x) in zip([e for e in entries if e[0] == "P2"],
                                                       [x for x in gt["zone_exits"] if x[0] == "P2"]))
    qs = [
        {"id": "S01", "question": "Which person entered the restricted area after the delivery truck arrived, and when?",
         "expected": {"subjects": ["P2"], "times": [p2_entries[0]]}, "skill": "order / after"},
        {"id": "S02", "question": "How many times did the machine stop?",
         "expected": {"value": 2, "times": [s for s, _ in MACHINE_STOPS]}, "skill": "counting over time"},
        {"id": "S03", "question": "What happened right before the safety alarm went off?",
         "expected": {"subjects": ["P2"], "times": [p2_entries[1]]}, "skill": "cause & effect / before"},
        {"id": "S04", "question": "When did the alarm go off?", "expected": {"times": [ALARM[0]]}, "skill": "event time"},
        {"id": "S05", "question": "When did the delivery vehicle arrive and how long did it stay parked?",
         "expected": {"times": [gt["vehicle_first_seen"]], "value": 50.0, "value_tol": 6.0}, "skill": "duration"},
        {"id": "S06", "question": "Did the first person who appeared come back later? If so, when?",
         "expected": {"subjects": ["P1"], "times": [40.0 + 12.0 * (0.08 / 1.14)]}, "skill": "re-identification"},
        {"id": "S07", "question": "Who stood still for more than 20 seconds, and when?",
         "expected": {"subjects": ["P3"], "times": [58.0]}, "skill": "stationary"},
        {"id": "S08", "question": "How many different people appear in the video?",
         "expected": {"value": 3}, "skill": "identity tracking"},
        {"id": "S09", "question": "In what order did people first enter the restricted area?",
         "expected": {"subjects": ["P1", "P2"], "times": [p1_entries[0], p2_entries[0]]}, "skill": "ordering"},
        {"id": "S10", "question": "Did anyone enter the restricted area while the machine was stopped?",
         "expected": {"value": "no"}, "skill": "temporal overlap"},
        {"id": "S11", "question": "How long in total did the second person spend inside the restricted area?",
         "expected": {"value": round(p2_time, 1), "value_tol": 2.5}, "skill": "duration aggregation"},
        {"id": "S12", "question": "When did a dog run across the floor?",
         "expected": {"not_observed": True}, "skill": "refuse unsupported"},
    ]
    meta = {"video": str(OUT_VIDEO.relative_to(ROOT)), "zones": [
        {"name": "Restricted area", "kind": "area", "points": ZONE},
        {"name": "Machine", "kind": "activity", "points": [(MACHINE[0], MACHINE[1]), (MACHINE[2], MACHINE[1]),
                                                           (MACHINE[2], MACHINE[3]), (MACHINE[0], MACHINE[3])]}],
            "ground_truth": gt, "time_tolerance": 1.5, "questions": qs}
    OUT_Q.parent.mkdir(exist_ok=True)
    OUT_Q.write_text(json.dumps(meta, indent=2))
    print("wrote", OUT_VIDEO, "and", OUT_Q)
    print(json.dumps(gt, indent=1))


def _ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


if __name__ == "__main__":
    main()
