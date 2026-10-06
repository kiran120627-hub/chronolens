"""Temporal question answering over the event log.

The LLM never watches the video and never invents a time. It reads the event log and identity table,
then writes a small pandas program `answer(...)` that computes the answer from those tables.
We execute that program (AST allow-list, restricted builtins, time limit), so every timestamp in the
answer is copied from detected events. If the evidence isn't there, the answer is "not observed".
"""
from __future__ import annotations

import ast
import json
import math
import re
import threading
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import fmt_t
from .llm import LLM, extract_json, extract_python
from .pipeline import Analysis

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "collections", "itertools", "datetime", "statistics"}
BANNED = {"open", "eval", "exec", "compile", "__import__", "input", "globals", "locals", "vars", "getattr", "setattr",
          "delattr", "exit", "quit", "breakpoint"}

SYSTEM = """You are ChronoLens, a temporal reasoning engine for video. You cannot see the video. You receive
the output of a deterministic perception pipeline (detector + tracker + re-identification + event engine) and must
answer questions about WHAT happened, WHEN, in WHICH ORDER, HOW OFTEN and FOR HOW LONG.

Hard rules
- Every answer must be computed by code from the tables. Never invent identities, events or times.
- Every claim needs timestamps (seconds) and the ids of the supporting events (column `id`, e.g. "E012").
- If the tables do not contain evidence for the answer, return found=False and say what was looked for.
  A wrong confident answer is worse than "not observed".
- Identities: P* = people, V* = vehicles, B* = bags (backpack/handbag/suitcase), O* = other objects; see ids.cls.
  The same id across gaps means the re-identification module linked them (see reid_score; lower = less sure).
- "Before X" means the latest relevant event that ends before X starts (look back a reasonable window, ~30 s,
  unless asked otherwise). "After X" means events starting after X starts.
- A "truck"/"delivery vehicle"/"car" in a question may map to any vehicle class (car, truck, bus).
- "Restricted area" or similar names map to the configured zone with the closest name; if there is exactly one area
  zone, use it.
- Counting "how many times" = number of distinct events of that type (after the engine's de-noising).
- For yes/no questions start the answer with "Yes" or "No" and also return value "yes"/"no".
- found=False ONLY when the kind of evidence needed does not exist in the tables (e.g. an animal that was never
  detected). A negative answer that IS supported by the tables ("No, nobody entered while it was stopped",
  "0 times") is found=True with the supporting events cited.

Tables (pandas DataFrames passed to your function)
- events: id, type, start, end, duration, subject, kind, cls, zone, other, details, confidence, when
  types: enter_view, exit_view, occluded, reappear, zone_enter, zone_exit, in_zone (interval), stationary (interval),
  untouched (interval, objects with no person contact), interaction (interval, subject=person, other=object/vehicle),
  activity_stop (interval: motion stopped inside an activity zone, e.g. a machine), activity_start,
  sudden_change (spike in motion / brightness / red light, e.g. a flashing alarm light),
  sound (from the audio track: details say "alarm-like tone / siren / beeper" or "loud sound (bang / crash / shout)"),
  caption (VLM keyframe description, if enabled)
- An "alarm" may be visual (sudden_change red light) or audible (sound alarm-like tone); consider both.
- ids: id, kind, cls, first_seen, last_seen, visible_seconds, tracklets, reid_links, reid_score
- tracks: per-sample boxes: t, gid, cls, conf, x1, y1, x2, y2 (pixels; frame size in meta)
- series: per-sample signals: t, motion, brightness, redness, act::<zone> (motion energy per activity zone)
- meta: dict with duration, fps, width, height, zones (list of {name, kind})
Helper available: fmt(t) -> "m:ss.s".

Reply with one ```json block {"plan": "1-3 sentences: how you will answer"} and one ```python block defining

    def answer(events, ids, tracks, series, meta):
        ...
        return {
            "found": True,                      # False if the evidence is not in the tables
            "answer": "plain-English answer that includes the key timestamps via fmt(...)",
            "value": <optional number/list for counts or durations>,
            "timestamps": [<seconds of the key moments, most important first>],
            "event_ids": ["E012", ...],         # supporting events
            "subjects": ["P2", ...],            # identities involved
        }

Allowed imports: pandas, numpy, math, re, collections, itertools. No file/network access, no printing needed."""


@dataclass
class QAResult:
    question: str
    status: str  # answered | not_observed | error
    answer: str = ""
    value: object = None
    timestamps: list = field(default_factory=list)
    event_ids: list = field(default_factory=list)
    subjects: list = field(default_factory=list)
    confidence: float = 0.0
    plan: str = ""
    engine: str = ""      # "LLM planner" | "rule engine"
    crosscheck: str = ""  # "agrees" | "overrode LLM" | ""
    code: str = ""
    attempts: int = 0
    error: str = ""
    evidence: list = field(default_factory=list)  # rows of cited events

    def to_dict(self) -> dict:
        return asdict(self)


def render_context(a: Analysis, max_events: int = 700) -> str:
    m = a.meta
    zones = [{"name": z.name, "kind": z.kind} for z in a.settings.zones]
    lines = [f"Video: {m['duration']:.1f}s, {m['fps']:.2f} fps, {m['width']}x{m['height']}, analysed at "
             f"{m['sample_fps']:.1f} samples/s. Zones: {json.dumps(zones)}"]
    ids = a.identities
    if len(ids):
        cols = [c for c in ["id", "kind", "cls", "first_seen", "last_seen", "visible_seconds", "reid_links", "reid_score"]
                if c in ids]
        lines.append("\n## Identities\n" + ids[cols].round(2).to_csv(index=False))
    ev = a.events
    counts = ev.type.value_counts().to_dict()
    lines.append(f"\n## Event counts\n{json.dumps(counts)}")
    cols = ["id", "type", "start", "end", "subject", "cls", "zone", "other", "details", "confidence"]
    if len(ev) > max_events:
        lines.append(f"\n## Events (first {max_events} of {len(ev)}; your code receives all of them)")
        lines.append(ev[cols].head(max_events).to_csv(index=False))
    else:
        lines.append("\n## Events\n" + ev[cols].to_csv(index=False))
    if len(a.links):
        lines.append("\n## Re-identification links\n" + a.links[["identity", "from_track", "to_track", "score", "gap"]]
                     .to_csv(index=False))
    return "\n".join(lines)


def static_check(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg} (line {e.lineno})"]
    bad = []
    if not any(isinstance(n, ast.FunctionDef) and n.name == "answer" for n in tree.body):
        bad.append("must define answer(events, ids, tracks, series, meta)")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bad += [f"import {a.name} not allowed" for a in n.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            bad.append(f"import from {n.module} not allowed")
        elif isinstance(n, ast.Name) and n.id in BANNED:
            bad.append(f"{n.id} is not allowed")
        elif isinstance(n, ast.Attribute) and (n.attr.startswith("__") or n.attr in ("to_csv", "to_pickle", "system")):
            bad.append(f"attribute {n.attr} not allowed")
    return bad


def _execute(code: str, a: Analysis, timeout: float = 20.0) -> dict:
    import builtins
    import collections
    import itertools

    safe_builtins = {k: getattr(builtins, k) for k in dir(builtins) if k not in BANNED and not k.startswith("_")}
    def _import(name, *args, **kw):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import of {name} not allowed")
        return __import__(name, *args, **kw)

    safe_builtins["__import__"] = _import
    g = {"__builtins__": safe_builtins, "pd": pd, "np": np, "math": math, "re": re, "collections": collections,
         "itertools": itertools, "fmt": fmt_t}
    exec(compile(code, "<answer>", "exec"), g)  # noqa: S102 - AST-checked, restricted builtins
    meta = {**a.meta, "zones": [{"name": z.name, "kind": z.kind} for z in a.settings.zones]}
    out: dict = {}
    err: list = []

    def run():
        try:
            out.update(g["answer"](a.events.copy(), a.identities.copy(), a.detections.copy(), a.series.copy(), meta))
        except Exception as e:  # noqa: BLE001
            import traceback

            err.append("".join(traceback.format_exception_only(type(e), e)).strip()
                       + "\n" + "\n".join(traceback.format_exc().splitlines()[-6:]))

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"answer() took longer than {timeout}s")
    if err:
        raise RuntimeError(err[0])
    if not isinstance(out, dict) or "answer" not in out:
        raise ValueError("answer() must return a dict with keys found, answer, timestamps, event_ids, subjects")
    return out


def _clean(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return [_clean(x) for x in list(v)]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _finish(a: Analysis, res: QAResult, out: dict) -> QAResult:
    found = bool(out.get("found", True))
    res.answer = str(out.get("answer", ""))
    res.value = _clean(out.get("value"))
    res.timestamps = [round(float(t), 2) for t in _clean(out.get("timestamps") or []) if t is not None][:12]
    valid_ids = set(a.events.id)
    res.event_ids = [e for e in [str(x) for x in _clean(out.get("event_ids") or [])] if e in valid_ids][:30]
    res.subjects = [str(s) for s in _clean(out.get("subjects") or [])][:20]
    res.status = "answered" if found else "not_observed"
    cited = a.events[a.events.id.isin(res.event_ids)]
    res.evidence = cited[["id", "type", "when", "subject", "zone", "other", "details", "confidence"]].to_dict("records")
    res.confidence = _confidence(a, res, cited, found)
    return res


def ask_offline(a: Analysis, question: str, reason: str = "") -> QAResult:
    from .offline import answer_offline

    res = QAResult(question=question, status="error", engine="rule engine",
                   plan="Offline rule engine over the event log (no LLM)" + (f" — {reason}" if reason else ""))
    out = answer_offline(a, question)
    if out is None:
        res.error = ("Offline mode understands counts, event times, before/after, ordering, loitering, untouched "
                     "objects, returns, sounds and time-in-zone questions — try rephrasing, or reconnect the LLM.")
        return res
    return _finish(a, res, out)


def ask(a: Analysis, question: str, llm: LLM | None, max_retries: int = 2, mode: str = "auto") -> QAResult:
    """mode: 'auto' (LLM, falls back to offline rules on failure) | 'llm' | 'offline'."""
    if mode == "offline" or llm is None or not llm.configured:
        return ask_offline(a, question, "" if mode == "offline" else "no LLM configured")
    try:
        res = _ask_llm(a, question, llm, max_retries)
    except Exception as e:  # noqa: BLE001 - network / quota failure
        if mode == "llm":
            raise
        return ask_offline(a, question, f"LLM unavailable: {str(e)[:120]}")
    if mode == "auto" and res.status == "answered":
        off = ask_offline(a, question, "cross-check")
        if off.status == "answered" and not _agree(res, off):
            off.plan = (f"Cross-check: the LLM's answer ({res.answer[:160]!r}) disagreed with the deterministic rule "
                        "engine, so the rule engine's evidence-backed answer is shown.")
            off.code = res.code
            off.crosscheck = "overrode LLM"
            return off
        if off.status == "answered":
            res.crosscheck = "agrees"
            res.confidence = round(min(1.0, res.confidence + 0.1), 2)
        return res
    if mode == "auto" and res.status in ("error", "not_observed"):
        # evidence beats refusal: if the deterministic rule engine finds supporting events, use them
        off = ask_offline(a, question, "LLM code failed" if res.status == "error" else "LLM found no evidence; "
                          "rule engine did")
        if off.status == "answered" or (res.status == "error" and off.status != "error"):
            return off
    return res


def _ask_llm(a: Analysis, question: str, llm: LLM, max_retries: int = 2) -> QAResult:
    res = QAResult(question=question, status="error", engine="LLM planner")
    ctx = render_context(a)
    msgs = [{"role": "user", "content": f"{ctx}\n\n## Question\n{question}"}]
    for attempt in range(max_retries + 1):
        res.attempts = attempt + 1
        text = llm.complete(SYSTEM, msgs, tag="qa")
        try:
            code = extract_python(text)
            try:
                res.plan = str(extract_json(text.split("```python")[0]).get("plan", ""))
            except (ValueError, AttributeError):
                res.plan = ""
            res.code = code
            problems = static_check(code)
            if problems:
                raise ValueError("static check failed: " + "; ".join(problems))
            out = _execute(code, a)
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Running your code failed:\n```\n{res.error[-2000:]}\n```\n"
                                                 "Fix it and reply again with the json and python blocks."}]
            continue
        res.error = ""
        return _finish(a, res, out)
    res.status = "error"
    return res


def _agree(x: QAResult, y: QAResult, tol: float = 1.5) -> bool:
    """Do two answers agree on the checkable facts (value and key timestamps)?"""
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    vx, vy = x.value, y.value
    if isinstance(vx, str) or isinstance(vy, str):
        if str(vx).lower() in ("yes", "no") and str(vy).lower() in ("yes", "no") and str(vx).lower() != str(vy).lower():
            return False
    elif num(vx) is not None and num(vy) is not None:
        if abs(num(vx) - num(vy)) > max(0.01 * abs(num(vy)), 0.5):
            return False
    elif num(vy) is not None and vx is None and y.timestamps and not x.timestamps:
        return False
    key = y.timestamps[:2]
    return all(any(abs(t - u) <= tol for u in x.timestamps) for t in key) if key else True


def _confidence(a: Analysis, res: QAResult, cited: pd.DataFrame, found: bool) -> float:
    if not found:
        return 0.0
    if cited.empty:
        return 0.35 if res.timestamps else 0.2
    c = float(cited.confidence.clip(0, 1).mean())
    ids = a.identities.set_index("id") if len(a.identities) else None
    if ids is not None:
        for s in set(res.subjects) | set(cited.subject.dropna()):
            if s in ids.index and int(ids.loc[s, "reid_links"]) > 0:
                c = min(c, float(ids.loc[s, "reid_score"]))
    # timestamps that don't correspond to any cited event lower trust
    starts = set(np.round(cited.start, 1)) | set(np.round(cited.end, 1))
    if res.timestamps and not any(round(t, 1) in starts for t in res.timestamps):
        c *= 0.8
    return round(c, 2)
