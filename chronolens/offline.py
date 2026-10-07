"""Offline answering: a rule engine over the event log for the most common temporal question types.

Used automatically when no LLM is configured or the LLM is unreachable / out of quota, and selectable in the UI.
It understands counting ("how many times…"), event times ("when did the alarm…"), ordering and "after/before X"
for zone entries, "right before X", loitering / untouched objects with duration thresholds, people returning,
loud sounds, and time spent in a zone. Anything else is answered honestly: it says what it cannot parse.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import fmt_t
from .pipeline import Analysis

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
UNSEEN = ["dog", "cat", "horse", "bird", "fire", "smoke", "gun", "knife", "weapon", "child", "bicycle", "motorbike",
          "forklift", "ladder", "helmet", "umbrella", "phone"]


def _duration(q: str, default: float) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|m\b|seconds?|secs?|s\b)", q)
    if not m:
        return default
    v = float(m.group(1))
    return v * 60 if m.group(2).startswith("m") else v


def _person(a: Analysis, q: str) -> str | None:
    m = re.search(r"\b(p\d+)\b", q)
    if m:
        return m.group(1).upper()
    people = a.identities[a.identities.kind == "person"].sort_values("first_seen")
    for word, n in ORDINALS.items():
        if re.search(rf"\b{word}\b (person|people|man|woman|worker|visitor)", q) and len(people) >= n:
            return people.iloc[n - 1].id
    return None


def _zone(a: Analysis, q: str) -> str | None:
    zones = [z.name for z in a.settings.zones if z.kind == "area"]
    for z in zones:
        if z.lower() in q:
            return z
    # only assume the default zone when the question actually talks about an area
    return zones[0] if zones and re.search(r"restrict|zone|area|forbidden|danger|inside|door|gate|entrance", q) else None


def _alarm(ev: pd.DataFrame) -> pd.DataFrame:
    a = ev[((ev.type == "sound") & ev.details.str.contains("alarm")) |
           ((ev.type == "sudden_change") & ev.details.str.contains("red|flash"))]
    return a.sort_values("start")


def _vehicle_arrival(ev: pd.DataFrame, a: Analysis) -> pd.Series | None:
    vids = set(a.identities[a.identities.kind == "vehicle"].id)
    arr = ev[(ev.type == "enter_view") & ev.subject.isin(vids)].sort_values("start")
    return arr.iloc[0] if len(arr) else None


def _result(found, answer, rows: pd.DataFrame | None = None, value=None, subjects=None, times=None) -> dict:
    rows = rows if rows is not None else pd.DataFrame(columns=["id", "start", "subject"])
    return {"found": found, "answer": answer, "value": value,
            "timestamps": list(times if times is not None else rows.start.round(2))[:12],
            "event_ids": list(rows.id)[:30],
            "subjects": subjects if subjects is not None else [s for s in dict.fromkeys(rows.subject) if s]}


def answer_offline(a: Analysis, question: str) -> dict | None:
    q = question.lower().strip()
    ev = a.events
    people = a.identities[a.identities.kind == "person"] if len(a.identities) else a.identities

    # things the pipeline cannot see at all
    for w in UNSEEN:
        if re.search(rf"\b{w}s?\b", q) and not (ev.type.eq("caption") & ev.details.str.lower().str.contains(w)).any() \
                and not a.identities.cls.astype(str).str.contains(w).any():
            return _result(False, f"Not observed: nothing matching '{w}' was detected in this video.")

    if re.search(r"how many .*(people|persons|individuals)|how many different", q):
        n = len(people)
        rows = ev[(ev.type == "enter_view") & ev.subject.isin(set(people.id))]
        return _result(True, f"{n} different people appear: " + ", ".join(
            f"{r.id} ({fmt_t(r.first_seen)}–{fmt_t(r.last_seen)})" for r in people.itertuples()) + ".", rows, value=n)

    if re.search(r"how many times.*(stop|halt)|how often.*(stop|halt)", q):
        rows = ev[ev.type == "activity_stop"]
        return _result(True, f"It stopped {len(rows)} time(s): " + ", ".join(f"{fmt_t(r.start)}–{fmt_t(r.end)}"
                                                                         for r in rows.itertuples()) + ".", rows, value=len(rows))

    alarm = _alarm(ev)
    if re.search(r"(right |just |immediately )?before .*alarm", q):
        if alarm.empty:
            return _result(False, "Not observed: no alarm (sound or flashing light) was detected.")
        t0 = alarm.start.iloc[0]
        prior = ev[(ev.start < t0 - 0.05) & (ev.start >= t0 - 30) &
                   ev.type.isin(["zone_enter", "zone_exit", "interaction", "enter_view", "reappear", "sound",
                                 "activity_stop", "activity_start"])].sort_values("start")
        if prior.empty:
            return _result(True, f"The alarm started at {fmt_t(t0)}; nothing notable happened in the 30 s before it.",
                           alarm.head(1))
        last = prior.iloc[-1]
        return _result(True, f"Right before the alarm at {fmt_t(t0)}: {last.details} at {fmt_t(last.start)} "
                             f"({t0 - last.start:.1f}s earlier).", pd.concat([prior.tail(1), alarm.head(1)]),
                       times=[last.start, t0], subjects=[last.subject] if last.subject else [])

    if re.search(r"\balarm\b", q):
        if alarm.empty:
            return _result(False, "Not observed: no alarm sound or flashing light was detected.")
        r0 = alarm.iloc[0]
        return _result(True, f"The alarm went off at {fmt_t(r0.start)} (until {fmt_t(alarm.end.max())}): "
                             + "; ".join(sorted(set(alarm.details))) + ".", alarm)

    if re.search(r"crash|bang|loud|noise|sound|shout", q):
        rows = ev[ev.type == "sound"]
        if re.search(r"crash|bang", q):
            rows = rows[rows.details.str.contains("loud sound")]
        if rows.empty:
            return _result(False, "Not observed: no loud sound was detected in the audio track.")
        return _result(True, "Loud sound(s) at " + ", ".join(f"{fmt_t(r.start)} ({r.details})" for r in rows.itertuples())
                       + ".", rows)

    if re.search(r"(when|what time).*(arriv|vehicle|truck|car|bus)", q):
        arr = _vehicle_arrival(ev, a)
        if arr is None:
            return _result(False, "Not observed: no vehicle was detected.")
        st = ev[(ev.type == "stationary") & (ev.subject == arr.subject)]
        extra = (f" It stayed parked {fmt_t(st.start.iloc[0])}–{fmt_t(st.end.iloc[0])} ({st.duration.iloc[0]:.1f}s)."
                 if len(st) else "")
        return _result(True, f"The vehicle {arr.subject} arrived at {fmt_t(arr.start)}.{extra}",
                       pd.concat([ev[ev.id == arr.id], st.head(1)]),
                       value=float(st.duration.iloc[0]) if len(st) else None)
    zone = _zone(a, q)
    if zone and re.search(r"order", q):
        firsts = ev[(ev.type == "zone_enter") & (ev.zone == zone)].sort_values("start").drop_duplicates("subject")
        return _result(not firsts.empty, "Order of first entry into " + zone + ": " + ", ".join(
            f"{r.subject} at {fmt_t(r.start)}" for r in firsts.itertuples()) + "." if len(firsts) else
            f"Nobody entered {zone}.", firsts)

    if zone and re.search(r"while .*(stop|stopped|off)", q):
        stops = ev[ev.type == "activity_stop"]
        ent = ev[(ev.type == "zone_enter") & (ev.zone == zone)]
        hits = ent[[any(s.start <= e.start <= s.end for s in stops.itertuples()) for e in ent.itertuples()]]
        if hits.empty:
            return _result(True, f"No. Nobody entered {zone} while the machine was stopped (stops: " + ", ".join(
                f"{fmt_t(s.start)}–{fmt_t(s.end)}" for s in stops.itertuples()) + ").", stops, value="no")
        return _result(True, "Yes: " + ", ".join(f"{r.subject} at {fmt_t(r.start)}" for r in hits.itertuples()) + ".",
                       hits, value="yes")

    if zone and re.search(r"how long|total time|time .*spen|spend", q):
        pid = _person(a, q)
        rows = ev[(ev.type == "in_zone") & (ev.zone == zone) & ((ev.subject == pid) if pid else True)]
        tot = float(rows.duration.sum())
        who = pid or "everyone"
        return _result(not rows.empty, f"{who} spent {tot:.1f}s in total inside {zone} across {len(rows)} visit(s): "
                       + ", ".join(f"{fmt_t(r.start)}–{fmt_t(r.end)}" for r in rows.itertuples()) + ".", rows,
                       value=round(tot, 1))

    if zone and re.search(r"enter|went into|walk(ed)? into|go into|inside|st(oo|a)d at|stopp?(ed)? at|went to|"
                          r"go to|approach|visit|at the (door|gate|zone|area)", q):
        rows = ev[(ev.type == "zone_enter") & (ev.zone == zone) & ev.subject.str.startswith("P")].sort_values("start")
        ref_txt = ""
        if re.search(r"after .*(truck|vehicle|delivery|car|bus|van)", q):
            arr = _vehicle_arrival(ev, a)
            if arr is not None:
                rows = rows[rows.start > arr.start]
                ref_txt = f" after the vehicle {arr.subject} arrived at {fmt_t(arr.start)}"
        elif re.search(r"before .*(truck|vehicle|delivery|car|bus|van)", q):
            arr = _vehicle_arrival(ev, a)
            if arr is not None:
                rows = rows[rows.start < arr.start]
                ref_txt = f" before the vehicle {arr.subject} arrived at {fmt_t(arr.start)}"
        if rows.empty:
            return _result(False, f"Nobody entered {zone}{ref_txt}.")
        first = rows.drop_duplicates("subject")
        return _result(True, f"Entered {zone}{ref_txt}: " + ", ".join(f"{r.subject} at {fmt_t(r.start)}"
                                                                      for r in rows.itertuples()) + ".", rows,
                       subjects=list(first.subject))

    if re.search(r"stood still|standing still|loiter|did ?n.?t move|not move|stationary|waited", q):
        n = _duration(q, a.settings.stationary_min)
        rows = ev[(ev.type == "stationary") & ev.subject.str.startswith("P") & (ev.duration > n)]
        if rows.empty:
            return _result(False, f"Nobody stood still for more than {n:.0f}s.")
        return _result(True, "Stood still >" + f"{n:.0f}s: " + ", ".join(
            f"{r.subject} from {fmt_t(r.start)} to {fmt_t(r.end)} ({r.duration:.1f}s)" for r in rows.itertuples()) + ".", rows)

    if re.search(r"who (left|dropped|put|placed)|whose bag|owner", q):
        rows = ev[ev.type == "object_left"]
        if rows.empty:
            return _result(False, "Not observed: no object was seen being left behind.")
        return _result(True, "; ".join(f"{r.details} at {fmt_t(r.start)}" for r in rows.itertuples()) + ".", rows,
                       subjects=list(dict.fromkeys(list(rows.other) + list(rows.subject))))

    if re.search(r"untouched|left behind|abandon|unattended|left (a|the|any) (bag|object)|(bag|object) .*left", q):
        n = _duration(q, a.settings.stationary_min)
        rows = ev[(ev.type == "untouched") & (ev.duration > n)]
        if rows.empty:
            return _result(False, f"No object sat untouched for more than {n:.0f}s.")
        return _result(True, "Untouched >" + f"{n:.0f}s: " + ", ".join(
            f"{r.subject} ({r.cls}) {fmt_t(r.start)}–{fmt_t(r.end)} ({r.duration:.0f}s)" for r in rows.itertuples()) + ".", rows)

    if re.search(r"come back|came back|return|reappear", q):
        pid = _person(a, q)
        rows = ev[(ev.type == "reappear") & ((ev.subject == pid) if pid else ev.subject.str.startswith("P"))]
        rows = rows[rows.details.str.contains(r"after (?:[3-9]|\d\d+)\.", regex=True)]  # real returns, not blinks
        if rows.empty:
            return _result(True, f"No — {pid or 'nobody'} did not come back after leaving.", value="no",
                           subjects=[pid] if pid else [])
        return _result(True, "Yes: " + ", ".join(f"{r.subject} came back at {fmt_t(r.start)} ({r.details})"
                                                 for r in rows.itertuples()) + ".", rows, value="yes")

    return None
