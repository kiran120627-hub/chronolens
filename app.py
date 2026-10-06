"""ChronoLens — Streamlit UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from chronolens.config import Settings, Zone, fmt_t
from chronolens.llm import LLM
from chronolens.pipeline import PALETTE, analyze, frame_at
from chronolens.qa import ask

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "videos" / "warehouse_sim.mp4"
SAMPLE_META = ROOT / "eval" / "questions_sim.json"
UPLOADS = ROOT / "data" / "uploads"

st.set_page_config(page_title="ChronoLens", page_icon="🎞️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.cl-hero h1 {font-size: 1.9rem; margin: 0; letter-spacing: -0.02em;}
.cl-hero p {color: #94a3b8; margin: .15rem 0 0 0;}
.cl-card {border: 1px solid rgba(148,163,184,.25); border-radius: 14px; padding: 1rem 1.2rem; margin: .3rem 0 .8rem 0;
          background: rgba(30,41,59,.35);}
.cl-badge {display:inline-block; font-weight:700; font-size:.72rem; letter-spacing:.05em; text-transform:uppercase;
           padding:.2rem .55rem; border-radius:999px;}
.cl-answered {background:#14532d; color:#bbf7d0;} .cl-not_observed {background:#312e81; color:#c7d2fe;}
.cl-error {background:#7f1d1d; color:#fecaca;}
.cl-answer {font-size: 1.25rem; font-weight: 600; margin: .5rem 0 .3rem 0; line-height: 1.45;}
.cl-meter {height:8px; background:rgba(148,163,184,.25); border-radius:999px; overflow:hidden; margin:.25rem 0 .1rem 0;}
.cl-meter > div {height:100%; border-radius:999px;}
.cl-ev {font-size:.88rem; color:#cbd5e1; padding:.25rem 0; border-bottom:1px dashed rgba(148,163,184,.2);}
.cl-ev code {color:#93c5fd;}
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "Which person entered the restricted area after the delivery truck arrived, and when?",
    "How many times did the machine stop?",
    "What happened right before the safety alarm went off?",
    "Did the first person who appeared come back later? If so, when?",
    "Who stood still for more than 20 seconds, and when?",
    "Did anyone enter the restricted area while the machine was stopped?",
    "How many different people appear in the video?",
    "When did a dog run across the floor?",
]

ss = st.session_state
ss.setdefault("seek", 0.0)
ss.setdefault("qa", None)
ss.setdefault("analysis", None)


# ----------------------------------------------------------------------------- helpers
def default_zones(video: Path) -> list[dict]:
    if video == SAMPLE and SAMPLE_META.exists():
        return json.loads(SAMPLE_META.read_text())["zones"]
    return [{"name": "Restricted area", "kind": "area", "points": [(0.6, 0.55), (0.98, 0.55), (0.98, 0.98), (0.6, 0.98)]}]


def rect_of(z: dict):
    xs, ys = [p[0] for p in z["points"]], [p[1] for p in z["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def first_frame(video: Path, zones: list[dict], width: int = 720):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = f.shape[:2]
    for i, z in enumerate(zones):
        col = (0, 0, 255) if z["kind"] == "area" else (0, 200, 255)
        pts = [(int(x * w), int(y * h)) for x, y in z["points"]]
        import numpy as np

        arr = np.array(pts, np.int32)
        ov = f.copy()
        cv2.fillPoly(ov, [arr], col)
        f = cv2.addWeighted(ov, 0.2, f, 0.8, 0)
        cv2.polylines(f, [arr], True, col, 3)
        cv2.putText(f, z["name"], (pts[0][0] + 6, pts[0][1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    f = cv2.resize(f, (width, int(h * width / w)))
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def seek(t: float):
    ss.seek = max(0.0, float(t) - 1.0)


# ----------------------------------------------------------------------------- sidebar
llm = LLM.from_env()
with st.sidebar:
    st.markdown("### 🎞️ ChronoLens")
    st.caption("Ask what happened, when, and in what order. Every answer is timestamped and backed by evidence.")
    src = st.radio("Video", ["Sample · warehouse (ground truth)", "Upload a video"], label_visibility="collapsed")
    video = SAMPLE
    if src == "Upload a video":
        up = st.file_uploader("MP4 / MOV / AVI / MKV", type=["mp4", "mov", "avi", "mkv", "webm"])
        if up:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            video = UPLOADS / f"{hashlib.sha1(up.getvalue()[:1 << 20]).hexdigest()[:10]}_{Path(up.name).name}"
            if not video.exists():
                video.write_bytes(up.getvalue())
    if ss.get("zones_for") != str(video):
        ss.zones = default_zones(video)
        ss.zones_for = str(video)
        ss.analysis, ss.qa = None, None

    with st.expander("Zones", expanded=src != "Sample · warehouse (ground truth)"):
        st.caption("Area zones log enter/exit. Activity zones watch for motion stopping (e.g. a machine).")
        for i, z in enumerate(ss.zones):
            x1, y1, x2, y2 = rect_of(z)
            c1, c2 = st.columns([2, 1])
            z["name"] = c1.text_input("name", z["name"], key=f"zn{i}", label_visibility="collapsed")
            z["kind"] = c2.selectbox("kind", ["area", "activity"], index=0 if z["kind"] == "area" else 1, key=f"zk{i}",
                                     label_visibility="collapsed")
            xr = st.slider("x range", 0.0, 1.0, (float(x1), float(x2)), 0.01, key=f"zx{i}")
            yr = st.slider("y range", 0.0, 1.0, (float(y1), float(y2)), 0.01, key=f"zy{i}")
            z["points"] = [(xr[0], yr[0]), (xr[1], yr[0]), (xr[1], yr[1]), (xr[0], yr[1])]
            if st.button("Remove zone", key=f"zr{i}"):
                ss.zones.pop(i)
                st.rerun()
            st.divider()
        if st.button("+ Add zone"):
            ss.zones.append({"name": f"Zone {len(ss.zones) + 1}", "kind": "area",
                             "points": [(0.3, 0.5), (0.6, 0.5), (0.6, 0.9), (0.3, 0.9)]})
            st.rerun()

    with st.expander("Analysis settings"):
        model = st.selectbox("Detector", ["yolo11s.pt", "yolo11n.pt", "yolo11m.pt"], help="n = fastest, m = most accurate")
        fps = st.slider("Analysis frame rate", 2, 25, 10)
        stationary = st.slider("Stationary after (s)", 3, 180, 8)
        reid = st.slider("Re-identification threshold", 0.3, 0.9, 0.62, 0.01)
        captions = st.slider("Caption a keyframe every N s (0 = off, uses the LLM)", 0, 30, 0)

    settings = Settings(target_fps=fps, model=model, stationary_min=stationary, reid_threshold=reid,
                        captions_every=float(captions),
                        zones=[Zone(z["name"], [tuple(p) for p in z["points"]], z["kind"]) for z in ss.zones])
    run = st.button("Analyse video", type="primary", use_container_width=True, disabled=not video.exists())
    st.divider()
    if llm.configured:
        st.success(llm.label, icon="🔑")
    else:
        st.warning("No LLM key — questions are answered by the offline rule engine", icon="⚠️")

# ----------------------------------------------------------------------------- header + run
st.markdown('<div class="cl-hero"><h1>ChronoLens</h1><p>Video understanding with temporal reasoning — detection, '
            'tracking & re-identification on the GPU, a deterministic event engine, and an LLM that can only answer '
            'from timestamped evidence.</p></div>', unsafe_allow_html=True)

if not video.exists():
    st.info("Upload a video in the sidebar (or generate the sample with `python data/make_synthetic.py`).")
    st.stop()

if run:
    bar = st.progress(0.0, "Starting…")
    try:
        ss.analysis = analyze(video, settings, progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m), llm=llm)
        ss.qa = None
        ss.seek = 0.0
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    bar.empty()

a = ss.analysis
if a is None:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        img = first_frame(video, ss.zones)
        if img is not None:
            st.image(img, caption="Zones preview — adjust them in the sidebar, then click Analyse video")
    with c2:
        st.markdown("#### How it works")
        st.markdown("1. **Detect & track** every person, vehicle and object (YOLO11 + BoT-SORT, GPU).\n"
                    "2. **Re-identify** people who were hidden or left and came back (appearance + motion).\n"
                    "3. **Event engine** turns tracks into exact, timestamped events: zone entries, stops, "
                    "loitering, interactions, alarms.\n"
                    "4. **Ask anything** — an LLM writes a query over the event log; the answer cites events, "
                    "timestamps and evidence frames, or says *not observed*.")
    st.stop()

# ----------------------------------------------------------------------------- results
m = a.meta
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Duration", fmt_t(m["duration"]))
k2.metric("People", int((a.identities.kind == "person").sum()) if len(a.identities) else 0)
k3.metric("Vehicles / objects", int((a.identities.kind != "person").sum()) if len(a.identities) else 0)
k4.metric("Events", len(a.events))
k5.metric("Re-ID links", len(a.links))

tab_ask, tab_tl, tab_ids, tab_ev, tab_bench = st.tabs(["Ask", "Timeline", "Identities", "Event log", "Benchmark"])

with tab_ask:
    left, right = st.columns([1.35, 1])
    with left:
        st.video(str(a.annotated), start_time=int(ss.seek), autoplay=ss.seek > 0, muted=True)
        st.caption(f"Annotated video · playing from {fmt_t(ss.seek)} · click any timestamp to jump")
    with right:
        st.caption("Try:")
        ex_cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if ex_cols[i % 2].button(ex if len(ex) < 52 else ex[:49] + "…", key=f"ex{i}", use_container_width=True):
                ss.question = ex
        with st.form("ask"):
            q = st.text_area("Question", value=ss.get("question", ""), height=70,
                             placeholder="e.g. What happened right before the alarm?")
            mode = st.radio("Answering", ["Auto · LLM + cross-check", "Offline rules (no internet)"], horizontal=True,
                            index=0 if llm.configured else 1, label_visibility="collapsed")
            go = st.form_submit_button("Ask", type="primary")
        if go and q.strip():
            ss.question = q
            with st.spinner("Reasoning over the event log…"):
                ss.qa = ask(a, q.strip(), llm, mode="offline" if mode.startswith("Offline") else "auto")
            if ss.qa.timestamps:
                seek(ss.qa.timestamps[0])
            st.rerun()

        r = ss.qa
        if r is not None:
            label = {"answered": "Answer", "not_observed": "Not observed in the video", "error": "Error"}[r.status]
            conf = int(r.confidence * 100)
            col = "#22c55e" if conf >= 70 else ("#eab308" if conf >= 40 else "#ef4444")
            body = (f'<span class="cl-badge cl-{r.status}">{label}</span>'
                    f'<div class="cl-answer">{html.escape(r.answer or r.error)}</div>')
            if r.status != "error":
                body += (f'<div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:.8rem">'
                         f'<span>Evidence confidence</span><span>{conf}%</span></div>'
                         f'<div class="cl-meter"><div style="width:{conf}%;background:{col}"></div></div>')
            if r.plan:
                body += f'<div style="color:#94a3b8;font-size:.8rem;margin-top:.5rem">⚙ {html.escape(r.plan[:300])}</div>'
            st.markdown(f'<div class="cl-card">{body}</div>', unsafe_allow_html=True)
            if r.timestamps:
                st.caption("Jump to:")
                tcols = st.columns(min(6, len(r.timestamps)))
                for i, t in enumerate(r.timestamps[:6]):
                    if tcols[i].button(f"▶ {fmt_t(t)}", key=f"ts{i}", use_container_width=True):
                        seek(t)
                        st.rerun()
            if r.evidence:
                rows = "".join(f'<div class="cl-ev"><code>{e["id"]}</code> <b>{html.escape(str(e["when"]))}</b> · '
                               f'{html.escape(str(e["details"]))} <span style="color:#64748b">({e["confidence"]:.2f})</span></div>'
                               for e in r.evidence[:12])
                st.markdown(f'<div class="cl-card"><b>Supporting events</b>{rows}</div>', unsafe_allow_html=True)
            if r.timestamps:
                frames = []
                for t in r.timestamps[:3]:
                    near = a.detections[(a.detections.t - t).abs() <= 0.6 / max(m["sample_fps"], 1) + 0.05]
                    if r.subjects:
                        near = near[near.gid.isin(r.subjects)]
                    near = near.sort_values("t").drop_duplicates("gid")
                    img = frame_at(a.video, t, 420, near, m)
                    if img is not None:
                        frames.append((t, img))
                if frames:
                    fcols = st.columns(len(frames))
                    for (t, img), c in zip(frames, fcols):
                        c.image(img, caption=f"evidence @ {fmt_t(t)}", use_container_width=True)
            with st.expander("How this answer was computed"):
                if r.plan:
                    st.write(r.plan)
                st.code(r.code or "", language="python")
                st.caption(f"{r.attempts} attempt(s) · model {llm.label}")

with tab_tl:
    import plotly.graph_objects as go

    fig = go.Figure()
    rows = list(a.segments.keys())
    for i, gid in enumerate(sorted(rows, key=lambda g: a.identities.set_index("id").loc[g, "first_seen"]
                                   if g in set(a.identities.id) else 0)):
        c = "rgb(%d,%d,%d)" % PALETTE[i % len(PALETTE)][::-1]
        cls = a.identities.set_index("id").loc[gid, "cls"] if gid in set(a.identities.id) else ""
        for s, e in a.segments[gid]:
            fig.add_trace(go.Bar(x=[max(e - s, 0.2)], y=[f"{gid} {cls}"], base=[s], orientation="h", marker_color=c,
                                 showlegend=False, hovertemplate=f"{gid} visible {fmt_t(s)}–{fmt_t(e)}<extra></extra>"))
    ev = a.events
    for typ, sym, colr in (("zone_enter", "triangle-right", "#ef4444"), ("zone_exit", "triangle-left", "#f97316"),
                           ("reappear", "star", "#a855f7")):
        sub = ev[ev.type == typ]
        if len(sub):
            ys = [f"{s} {a.identities.set_index('id').loc[s, 'cls']}" if s in set(a.identities.id) else s for s in sub.subject]
            fig.add_trace(go.Scatter(x=sub.start, y=ys, mode="markers", marker=dict(symbol=sym, size=13, color=colr),
                                     name=typ, text=sub.details, hovertemplate="%{text} @ %{x:.1f}s<extra></extra>"))
    for typ, label, colr in (("activity_stop", "machine / activity stopped", "rgba(234,179,8,.35)"),
                             ("sudden_change", "sudden change / alarm", "rgba(239,68,68,.35)")):
        for r_ in ev[ev.type == typ].itertuples():
            fig.add_vrect(x0=r_.start, x1=max(r_.end, r_.start + 0.4), fillcolor=colr, line_width=0,
                          annotation_text=label if r_.Index == ev[ev.type == typ].index[0] else "",
                          annotation_position="top left")
    fig.update_layout(height=140 + 38 * len(rows), barmode="overlay", xaxis_title="seconds",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars = when each identity is visible (gaps = hidden/out of frame). Markers = zone entries/exits and "
               "re-identifications. Shaded = machine stops (yellow) and sudden changes such as alarms (red).")

with tab_ids:
    ids = a.identities
    if ids.empty:
        st.info("No identities detected.")
    else:
        cols = st.columns(6)
        for i, row in enumerate(ids.itertuples()):
            with cols[i % 6]:
                p = a.thumb(row.id)
                if p:
                    st.image(str(p), use_container_width=True)
                st.markdown(f"**{row.id}** · {row.cls}")
                st.caption(f"{row.first} → {row.last} · visible {row.visible_seconds:.0f}s"
                           + (f" · {row.reid_links} re-ID link(s), score {row.reid_score:.2f}" if row.reid_links else ""))
        if len(a.links):
            st.markdown("#### Re-identification decisions")
            st.dataframe(a.links, hide_index=True, use_container_width=True)

with tab_ev:
    types = sorted(a.events.type.unique())
    pick = st.multiselect("Event types", types, default=[t for t in types if t not in ("in_zone",)])
    view = a.events[a.events.type.isin(pick)][["id", "when", "type", "subject", "zone", "other", "details", "confidence"]]
    sel = st.dataframe(view, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                       height=520)
    if sel and sel.selection.rows:
        row = view.iloc[sel.selection.rows[0]]
        t0 = float(a.events.set_index("id").loc[row.id, "start"])
        st.button(f"▶ Play from {fmt_t(t0)} (switch to the Ask tab)", on_click=seek, args=(t0,))
    st.download_button("Download event log (CSV)", a.events.to_csv(index=False), "events.csv", "text/csv")

with tab_bench:
    rep = ROOT / "eval" / "REPORT.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.info("Run `python eval/run_eval.py` to score ChronoLens on the ground-truth benchmark video.")
