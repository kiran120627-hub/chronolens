"""Tunable parameters, zones and shared helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# COCO classes we track. Everything else is ignored.
TRACK_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    24: "backpack", 26: "handbag", 28: "suitcase", 39: "bottle", 63: "laptop", 67: "cell phone",
}
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
CARRIED_OBJECTS = {"backpack", "handbag", "suitcase", "bottle", "laptop", "cell phone"}


@dataclass
class Zone:
    """Polygon in normalised coordinates (0..1). kind = 'area' (entry/exit) or 'activity' (motion on/off, e.g. a machine)."""
    name: str
    points: list[tuple[float, float]]
    kind: str = "area"

    @classmethod
    def rect(cls, name: str, x1: float, y1: float, x2: float, y2: float, kind: str = "area") -> "Zone":
        return cls(name, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], kind)


@dataclass
class Settings:
    target_fps: float = 10.0          # analysis sampling rate
    model: str = "yolo11s.pt"         # detector (n = fastest, s = balanced, m = most accurate)
    conf: float = 0.20                # detection confidence threshold (weak short tracks are filtered later)
    imgsz: int = 1280                 # inference size; larger finds small/far people and bags
    track_buffer: int = 60            # tracker frames to keep a lost track alive (60 @10fps = 6 s)
    min_track_seconds: float = 0.6    # ignore flickering tracks shorter than this
    reid_threshold: float = 0.62      # similarity that always links two tracklets into one identity
    reid_floor: float = 0.50          # ...or this much, if it is clearly the best candidate (by reid_margin)
    reid_margin: float = 0.12
    static_conf: float = 0.03         # sensitive bag detector for left-behind objects (validated by persistence)
    static_min_seconds: float = 5.0   # a weak detection must stay put this long to count as a real object
    reid_verify_low: float = 0.40     # grey zone [verify_low, threshold): ask a vision-language model (if configured)
    reid_verify_max: int = 8          # at most this many VLM checks per video
    reid_max_gap: float = 180.0       # max seconds between tracklets for re-identification
    segment_gap: float = 1.0          # gaps shorter than this are bridged when building visibility segments
    zone_min_dwell: float = 0.5       # hysteresis for zone enter/exit
    stationary_min: float = 8.0       # seconds without movement to count as stationary
    stationary_tol: float = 0.25      # max centre movement, as a fraction of the object's height
    activity_min_stop: float = 1.5    # min seconds of no motion in an activity zone to count as a stop
    interaction_margin: float = 0.25  # box expansion (fraction of object size) for person-object interaction
    render_width: int = 960           # width of the annotated output video
    captions_every: float = 0.0       # >0: caption a keyframe every N seconds with a vision-language model
    zones: list[Zone] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        d = dict(d)
        d["zones"] = [Zone(z["name"], [tuple(p) for p in z["points"]], z.get("kind", "area")) for z in d.get("zones", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def fmt_t(t: float | None) -> str:
    """Seconds -> 'm:ss.s' (or 'h:mm:ss')."""
    if t is None:
        return "—"
    t = max(0.0, float(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h else f"{int(m)}:{s:04.1f}"


def point_in_poly(x: float, y: float, pts: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside
