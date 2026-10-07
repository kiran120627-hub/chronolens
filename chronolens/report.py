"""Incident report: a self-contained, printable HTML summary of an analysed video.

Contains the key findings (derived deterministically from the event log), every question the investigator asked
with its answer, timestamps and evidence frames, and an appendix of the relevant events. Open it in a browser and
use Print -> Save as PDF for a PDF.
"""
from __future__ import annotations

import base64
import html
from datetime import datetime

import cv2

from .config import fmt_t
from .pipeline import Analysis, frame_at

CSS = """
*{box-sizing:border-box} body{font-family:Inter,Segoe UI,system-ui,sans-serif;color:#1d1f23;margin:0;background:#f4f4f2}
.page{max-width:900px;margin:0 auto;background:#fff;padding:44px 52px 60px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.02em} h2{font-size:15px;margin:34px 0 10px;text-transform:uppercase;
letter-spacing:.08em;color:#6b6e75;border-bottom:1px solid #e6e5e1;padding-bottom:6px}
.sub{color:#6b6e75;font-size:13px} .mono{font-family:Consolas,ui-monospace,monospace;font-variant-numeric:tabular-nums}
.stats{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid #e6e5e1;border-radius:10px;margin-top:22px}
.stats div{padding:12px 14px;border-right:1px solid #e6e5e1} .stats div:last-child{border-right:none}
.stats .l{font-size:11px;color:#8a8d93} .stats .v{font-size:20px;margin-top:2px}
ul.find{padding-left:18px;margin:0} ul.find li{margin:6px 0;line-height:1.5}
.qa{border:1px solid #e6e5e1;border-left:3px solid #4f9b74;border-radius:8px;padding:14px 16px;margin:12px 0;break-inside:avoid}
.qa.not_observed{border-left-color:#6f8fb8} .qa.error{border-left-color:#c4594e}
.qa .q{font-weight:600;margin-bottom:6px} .qa .a{line-height:1.55} .qa .m{font-size:12px;color:#6b6e75;margin-top:8px}
.frames{display:flex;gap:10px;margin-top:10px;flex-wrap:wrap} .frames figure{margin:0;width:31%}
.frames img{width:100%;border-radius:6px;border:1px solid #e6e5e1} .frames figcaption{font-size:11px;color:#6b6e75}
table{width:100%;border-collapse:collapse;font-size:12.5px} th{text-align:left;color:#8a8d93;font-weight:500;
border-bottom:1px solid #d9d8d3;padding:6px} td{border-bottom:1px solid #eeede9;padding:6px;vertical-align:top}
.foot{margin-top:40px;font-size:11.5px;color:#8a8d93;line-height:1.5}
@media print{body{background:#fff}.page{padding:0}}
"""


def _img(a: Analysis, t: float, subjects: list[str]) -> str:
    m = a.meta
    near = a.detections[(a.detections.t - t).abs() <= 0.6 / max(m["sample_fps"], 1) + 0.05]
    if subjects:
        near = near[near.gid.isin(subjects)]
    rgb = frame_at(a.video, t, 420, near.sort_values("t").drop_duplicates("gid"), m)
    if rgb is None:
        return ""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}" if ok else ""


def key_findings(a: Analysis) -> list[str]:
    """Plain-language findings, straight from the event log (no AI involved)."""
    ev, out = a.events, []
    e = lambda s: html.escape(str(s))  # noqa: E731
    for r in ev[ev.type == "object_left"].itertuples():
        un = ev[(ev.type == "untouched") & (ev.subject == r.subject)]
        dur = f", untouched for {un.duration.max():.0f} s" if len(un) else ""
        out.append(f"<b>Object left behind:</b> {e(r.subject)} ({e(r.cls)}) left by {e(r.other)} at "
                   f"<span class='mono'>{fmt_t(r.start)}</span>{dur}.")
    for r in ev[ev.type == "in_zone"].itertuples():
        out.append(f"<b>Zone entry:</b> {e(r.subject)} inside <i>{e(r.zone)}</i> from "
                   f"<span class='mono'>{fmt_t(r.start)}</span> to <span class='mono'>{fmt_t(r.end)}</span> "
                   f"({r.duration:.1f} s).")
    for r in ev[(ev.type == "stationary") & ev.subject.astype(str).str.startswith("P")].itertuples():
        out.append(f"<b>Loitering:</b> {e(r.subject)} did not move from <span class='mono'>{fmt_t(r.start)}</span> "
                   f"to <span class='mono'>{fmt_t(r.end)}</span> ({r.duration:.0f} s).")
    for r in ev[ev.type == "reappear"].itertuples():
        if str(r.subject).startswith("P") and "after" in str(r.details):
            try:
                gap = float(str(r.details).split("after ")[1].split("s")[0])
            except (IndexError, ValueError):
                gap = 0
            if gap >= 3:
                out.append(f"<b>Returned:</b> {e(r.subject)} came back at <span class='mono'>{fmt_t(r.start)}</span> "
                           f"after {gap:.0f} s out of view (re-identified).")
    stops = ev[ev.type == "activity_stop"]
    if len(stops):
        out.append(f"<b>Equipment stopped {len(stops)} time(s):</b> " + ", ".join(
            f"<span class='mono'>{fmt_t(r.start)}–{fmt_t(r.end)}</span>" for r in stops.itertuples()) + ".")
    alarms = ev[((ev.type == "sound") & ev.details.str.contains("alarm")) |
                ((ev.type == "sudden_change") & ev.details.str.contains("red"))]
    if len(alarms):
        out.append("<b>Alarm signals:</b> " + ", ".join(
            f"<span class='mono'>{fmt_t(r.start)}</span> ({e(r.details)})" for r in alarms.itertuples()) + ".")
    return out or ["No notable events were detected."]


def build_report(a: Analysis, video_name: str, qa_history: list, include_frames: bool = True) -> str:
    m, ids = a.meta, a.identities
    people = int((ids.kind == "person").sum()) if len(ids) else 0
    objects = int((ids.kind != "person").sum()) if len(ids) else 0
    stats = [("Duration", fmt_t(m["duration"])), ("People", people), ("Vehicles / objects", objects),
             ("Events", len(a.events)), ("Questions", len(qa_history))]
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Incident report — {html.escape(video_name)}</title>"
             f"<style>{CSS}</style></head><body><div class='page'>",
             "<div class='sub'>ChronoLens · incident report</div>",
             f"<h1>{html.escape(video_name)}</h1>",
             f"<div class='sub'>Generated {datetime.now().strftime('%d %b %Y, %H:%M')} · analysed with "
             f"{html.escape(str(m.get('model', '')))} on {html.escape(str(m.get('gpu') or m.get('device', '')))}</div>",
             "<div class='stats'>" + "".join(f"<div><div class='l'>{k}</div><div class='v mono'>{v}</div></div>"
                                             for k, v in stats) + "</div>",
             "<h2>Key findings</h2><ul class='find'>" + "".join(f"<li>{f}</li>" for f in key_findings(a)) + "</ul>"]

    if qa_history:
        parts.append("<h2>Investigation</h2>")
        for r in qa_history:
            label = {"answered": "Answer", "not_observed": "Not observed", "error": "Could not answer"}.get(r.status, "")
            meta = [f"{label}", f"confidence {int(r.confidence * 100)}%", r.engine or ""]
            if r.crosscheck == "agrees":
                meta.append("cross-checked by rule engine")
            ts = " · ".join(fmt_t(t) for t in r.timestamps[:6])
            if ts:
                meta.append(f"times {ts}")
            frames = ""
            if include_frames and r.timestamps:
                imgs = [(t, _img(a, t, r.subjects)) for t in r.timestamps[:3]]
                frames = "<div class='frames'>" + "".join(
                    f"<figure><img src='{src}' alt='Frame at {fmt_t(t)}'><figcaption class='mono'>{fmt_t(t)}"
                    f"</figcaption></figure>" for t, src in imgs if src) + "</div>"
            parts.append(f"<div class='qa {r.status}'><div class='q'>{html.escape(r.question)}</div>"
                         f"<div class='a'>{html.escape(r.answer or r.error)}</div>"
                         f"<div class='m'>{html.escape(' · '.join(x for x in meta if x))}</div>{frames}</div>")

    key = a.events[~a.events.type.isin(["in_zone", "caption"])]
    rows = "".join(f"<tr><td class='mono'>{r.id}</td><td class='mono'>{html.escape(str(r.when))}</td>"
                   f"<td>{html.escape(str(r.type))}</td><td>{html.escape(str(r.subject or ''))}</td>"
                   f"<td>{html.escape(str(r.details))}</td></tr>" for r in key.itertuples())
    parts.append("<h2>Event log</h2><table><tr><th>ID</th><th>Time</th><th>Type</th><th>Who</th><th>Details</th></tr>"
                 f"{rows}</table>")
    parts.append("<div class='foot'>Times come from a deterministic event log built from per-frame detections "
                 f"({m['sample_fps']:.1f} samples per second); no timestamp in this report was generated by a language "
                 "model. Identities that left and returned were linked by appearance and, where uncertain, "
                 "confirmed by a vision model. Event IDs refer to the exported event log.</div>")
    parts.append("</div></body></html>")
    return "".join(parts)


def report_filename(video_name: str) -> str:
    stem = "".join(c if c.isalnum() else "_" for c in video_name.rsplit(".", 1)[0])
    return f"incident_report_{stem}_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

