"""Audio events: loud sounds and alarm-like tones from the video's soundtrack.

Real safety alarms are usually *sounds*. We decode the audio track (ffmpeg -> 16 kHz mono), compute loudness
in 100 ms windows, and flag windows far above the clip's own background level. A window whose energy is
concentrated in a few narrow frequency bands (a siren / beeper) is labelled "alarm-like tone"; otherwise
"loud sound" (bang, shout, crash). Thresholds adapt to each clip, so quiet rooms and noisy factories both work.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000
WIN = 0.1  # seconds


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def load_audio(video: Path) -> np.ndarray | None:
    p = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                       capture_output=True)
    if p.returncode != 0 or len(p.stdout) < SR * 0.5 * 2:
        return None  # no audio track
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def audio_series(video: Path) -> pd.DataFrame:
    x = load_audio(video)
    if x is None:
        return pd.DataFrame(columns=["t", "db", "tonality"])
    n = int(SR * WIN)
    frames = len(x) // n
    x = x[: frames * n].reshape(frames, n)
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    spec = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / SR)
    band = (freqs >= 300) & (freqs <= 5000)
    sb = spec[:, band]
    # tonality: share of band energy in the 3 strongest bins (pure tones/beeps -> high, noise/speech -> low)
    top = np.sort(sb, axis=1)[:, -3:].sum(axis=1)
    tonality = top / (sb.sum(axis=1) + 1e-12)
    return pd.DataFrame({"t": np.round(np.arange(frames) * WIN, 2), "db": db, "tonality": tonality})


def audio_events(series: pd.DataFrame, min_rise_db: float = 12.0) -> list[dict]:
    """Return event dicts (type 'sound') for loud segments, labelled alarm-like vs loud sound."""
    if series.empty:
        return []
    db = series.db.to_numpy()
    base = float(np.median(db))
    mad = float(np.median(np.abs(db - base))) + 1e-6
    thr = base + max(min_rise_db, 4 * mad)
    loud = db > thr
    events, i = [], 0
    t = series.t.to_numpy()
    while i < len(loud):
        if not loud[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(loud) and (loud[j + 1] or (j + 2 < len(loud) and loud[j + 2])):  # bridge 1-window gaps
            j += 1
        dur = t[j] - t[i] + WIN
        if dur >= 0.2:
            seg = series.iloc[i:j + 1]
            tonal = float(seg.tonality.median())
            peak = float(seg.db.max())
            label = "alarm-like tone / siren / beeper" if tonal > 0.35 else "loud sound (bang / crash / shout)"
            events.append({"type": "sound", "start": round(float(t[i]), 2), "end": round(float(t[j] + WIN), 2),
                           "duration": round(dur, 2), "subject": "", "kind": "", "cls": "", "zone": "audio",
                           "other": "", "details": f"{label}, {peak - base:+.0f} dB above background",
                           "confidence": round(float(min(1.0, 0.55 + (peak - thr) / 30)), 2)})
        i = j + 1
    return events
