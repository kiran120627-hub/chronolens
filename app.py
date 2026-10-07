"""ChronoLens — Streamlit UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from chronolens.config import Settings, Zone, fmt_t
from chronolens.llm import LLM
from chronolens.pipeline import PALETTE, analyze, export_clip, frame_at
from chronolens.qa import ask

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "videos" / "warehouse_sim.mp4"
SAMPLE_META = ROOT / "eval" / "questions_sim.json"
UPLOADS = ROOT / "data" / "uploads"

st.set_page_config(page_title="ChronoLens", page_icon=str(ROOT / "docs" / "favicon.png")
                   if (ROOT / "docs" / "favicon.png").exists() else None, layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- design system
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');
:root {
  --bg: #0d0e10; --panel: #141518; --panel-2: #191b1f; --panel-3: #1f2226;
  --line: rgba(255,255,255,.06); --line-2: rgba(255,255,255,.11);
  --text: #e8e6e1; --muted: #9a9ca1; --dim: #64676d;
  --accent: #d6a250; --accent-soft: rgba(214,162,80,.12);
  --ok: #6dbb8f; --bad: #d46a5f; --info: #8aa6c9;
}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp div, .stApp button,
.stApp input, .stApp textarea, .stApp td, .stApp th { font-family: 'Geist', system-ui, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(214,162,80,.05), transparent 60%), var(--bg); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { max-width: 1400px; padding: 1.1rem 2.2rem 4rem; }
.mono, code { font-family: 'Geist Mono', ui-monospace, monospace !important; font-variant-numeric: tabular-nums; }

/* sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div { padding-top: .6rem; }
.cl-side-brand { display:flex; align-items:center; gap:.6rem; font-weight:600; font-size:1.02rem; letter-spacing:-.01em;
                 color: var(--text); padding: .35rem 0 1.1rem; }
.cl-mark { width: 22px; height: 22px; border-radius: 6px; background: var(--accent); position: relative; flex: none; }
.cl-mark::before, .cl-mark::after { content:""; position:absolute; background: var(--bg); border-radius: 1px; }
.cl-mark::before { left: 5px; top: 6px; width: 12px; height: 2px; }
.cl-mark::after  { left: 5px; top: 12px; width: 7px; height: 2px; }
.cl-label { font-size: .7rem; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: var(--dim);
            margin: 1.1rem 0 .45rem; }
.cl-zone { display:flex; align-items:center; justify-content:space-between; padding:.5rem .6rem; border-radius:8px;
           background: var(--panel-2); margin-bottom:.35rem; font-size:.86rem; }
.cl-zone .k { font-family:'Geist Mono',monospace; font-size:.7rem; color: var(--muted); border:1px solid var(--line-2);
              padding:.05rem .35rem; border-radius:4px; }
.cl-keyline { font-size:.78rem; color: var(--muted); display:flex; gap:.5rem; align-items:center; }
.dot { width:7px; height:7px; border-radius:50%; display:inline-block; background: var(--dim); }
.dot.ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(109,187,143,.15); }
.dot.warn { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }

/* top bar */
.cl-top { display:flex; align-items:center; justify-content:space-between; gap: 1rem; flex-wrap: wrap; padding: .2rem 0 1rem;
          border-bottom: 1px solid var(--line); margin-bottom: 1.1rem; }
.cl-top h1 { font-size: 1.32rem; font-weight: 600; letter-spacing: -.02em; margin: 0; color: var(--text); }
.cl-top .sub { color: var(--muted); font-size: .86rem; margin-top: .15rem; }
.cl-chips { display:flex; gap:.5rem; flex-wrap: wrap; justify-content:flex-end; }
.cl-chip { white-space: nowrap; }
.cl-chip { font-size:.75rem; color: var(--muted); border:1px solid var(--line-2); border-radius:6px; padding:.25rem .55rem;
           display:inline-flex; align-items:center; gap:.4rem; background: var(--panel); }
.cl-chip b { color: var(--text); font-weight: 500; }

/* stat strip */
.cl-stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 1px; background: var(--line);
            border:1px solid var(--line); border-radius: 12px; margin-bottom: 1.2rem; overflow:hidden; }
.cl-stat { padding: .85rem 1.1rem; background: var(--panel); min-width: 0; }
.cl-stat .l { font-size: .72rem; color: var(--dim); font-weight: 500; letter-spacing: .02em; white-space: nowrap;
              overflow: hidden; text-overflow: ellipsis; }
.cl-stat .v { white-space: nowrap; }
.cl-stat .v { font-family:'Geist Mono',monospace; font-variant-numeric: tabular-nums; font-size: 1.35rem; color: var(--text);
              margin-top: .15rem; letter-spacing: -.02em; }
.cl-stat .v small { font-size: .78rem; color: var(--muted); margin-left: .2rem; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: .55rem 0; font-weight: 500; color: var(--muted); background: transparent; }
.stTabs [aria-selected="true"] { color: var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* buttons & inputs */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 8px; border: 1px solid var(--line-2); background: var(--panel-2); color: var(--text); font-weight: 500;
  transition: background .18s ease, border-color .18s ease, transform .08s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: rgba(214,162,80,.55); background: var(--panel-3); color: var(--text); }
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  background: var(--accent); border-color: var(--accent); color: #17130c; font-weight: 600; }
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover { background: #e0b066; border-color: #e0b066; color: #17130c; }
textarea, input, [data-baseweb="select"] > div { background: var(--panel-2) !important; border-radius: 8px !important; }
[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); padding: 1rem 1rem .4rem; }
[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
[data-testid="stExpander"] summary { font-weight: 500; color: var(--muted); }
.stProgress > div > div > div > div { background: var(--accent); }
[data-testid="stPills"] button, [data-testid="stButtonGroup"] button { border-radius: 6px !important; font-size: .8rem !important; }
.stVideo, video { border-radius: 10px; }
img { border-radius: 8px; }

/* section headers */
.cl-h { display:flex; align-items:baseline; justify-content:space-between; margin: .2rem 0 .6rem; }
.cl-h h3 { font-size: .95rem; font-weight: 600; margin: 0; color: var(--text); letter-spacing: -.005em; }
.cl-h span { font-size: .78rem; color: var(--dim); }

/* answer */
.cl-answer { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1.1rem 1.25rem 1rem;
             position: relative; overflow: hidden; }
.cl-answer::before { content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--ok); }
.cl-answer.not_observed::before { background: var(--info); }
.cl-answer.error::before { background: var(--bad); }
.cl-status { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--ok); }
.cl-answer.not_observed .cl-status { color: var(--info); } .cl-answer.error .cl-status { color: var(--bad); }
.cl-answer p.a { font-size: 1.08rem; line-height: 1.55; color: var(--text); margin: .45rem 0 .9rem; text-wrap: pretty; max-width: 62ch; }
.cl-meta { display:grid; grid-template-columns: max-content minmax(0, 1fr); gap: .35rem .9rem; font-size: .8rem; color: var(--muted);
           border-top: 1px solid var(--line); padding-top: .75rem; }
.cl-meta .k { color: var(--dim); }
.cl-bar { height: 4px; background: var(--line-2); border-radius: 2px; overflow: hidden; width: min(160px, 50%); display:inline-block; vertical-align: middle; margin-left: .5rem; }
.cl-bar > i { display:block; height: 100%; background: var(--ok); }
.cl-warn { color: var(--accent); }
.cl-ev-row { display:grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: .15rem .7rem; padding: .55rem .2rem;
             font-size: .84rem; border-bottom: 1px solid var(--line); color: var(--muted); align-items: baseline; }
.cl-ev-row .d { grid-column: 1 / -1; line-height: 1.45; }
.cl-ev-row:last-child { border-bottom: none; }
.cl-ev-row .id { font-family:'Geist Mono',monospace; color: var(--dim); font-size: .76rem; }
.cl-ev-row .t { font-family:'Geist Mono',monospace; color: var(--text); font-variant-numeric: tabular-nums; }
.cl-ev-row .c { font-family:'Geist Mono',monospace; color: var(--dim); text-align: right; font-size: .76rem; }
.cl-box { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .5rem .9rem; }

/* empty state / steps */
.cl-steps { display:grid; gap: .9rem; }
.cl-step { display:grid; grid-template-columns: 34px 1fr; gap: .6rem; }
.cl-step .n { font-family:'Geist Mono',monospace; color: var(--accent); font-size: .8rem; padding-top: .1rem; }
.cl-step b { color: var(--text); font-weight: 500; display:block; margin-bottom: .1rem; }
.cl-step span { color: var(--muted); font-size: .86rem; line-height: 1.5; }

/* identity cards */
.cl-id-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: .8rem; margin-bottom: 1.4rem; }
.cl-card-id { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: .6rem; }
.cl-card-id .th { height: 170px; background: var(--panel-2); border-radius: 8px; display:flex; align-items:center;
                  justify-content:center; overflow:hidden; margin-bottom: .6rem; }
.cl-card-id .th img { max-height: 100%; max-width: 100%; object-fit: contain; border-radius: 0; }
.cl-card-id .hd { display:flex; align-items:center; gap:.45rem; margin: 0 .15rem .4rem; }
.cl-card-id .hd i { width:8px; height:8px; border-radius:2px; display:inline-block; }
.cl-card-id .hd b { font-family:'Geist Mono',monospace; font-weight:500; color: var(--text); }
.cl-card-id .hd span { color: var(--muted); font-size:.8rem; }
.cl-card-id .r { display:flex; justify-content:space-between; font-size:.76rem; color: var(--dim); padding: .18rem .15rem; }
.cl-card-id .r .mono { color: var(--muted); }
.cl-id { font-size: .82rem; color: var(--muted); margin-top: .35rem; line-height: 1.5; }
.cl-id b { font-family:'Geist Mono',monospace; color: var(--text); font-weight: 500; font-size: .9rem; }

/* markdown tables (benchmark) */
.stMarkdown table { border-collapse: collapse; width: 100%; font-size: .84rem; }
.stMarkdown th { color: var(--dim); font-weight: 500; text-align: left; border-bottom: 1px solid var(--line-2) !important; }
.stMarkdown td, .stMarkdown th { border: none !important; border-bottom: 1px solid var(--line) !important; padding: .45rem .6rem !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = {
    "Entry after truck": "Which person entered the restricted area after the delivery truck arrived, and when?",
    "Machine stops": "How many times did the machine stop?",
    "Before the alarm": "What happened right before the safety alarm went off?",
    "Return visit": "Did the first person who appeared come back later? If so, when?",
    "Loitering > 20 s": "Who stood still for more than 20 seconds, and when?",
    "Entry during stop": "Did anyone enter the restricted area while the machine was stopped?",
    "Head count": "How many different people appear in the video?",
    "Loud sounds": "Was there any loud crash or bang? When?",
    "Unseen object": "When did a dog run across the floor?",
}
# identity colours, matching the boxes in the annotated video (PALETTE is BGR)
ID_COLOURS = ["#%02x%02x%02x" % (c[2], c[1], c[0]) for c in PALETTE]

ss = st.session_state
ss.setdefault("seek", 0.0)
ss.setdefault("qa", None)
ss.setdefault("analysis", None)


# ----------------------------------------------------------------------------- helpers
def default_zones(video: Path) -> list[dict]:
    if video == SAMPLE and SAMPLE_META.exists():
        return json.loads(SAMPLE_META.read_text())["zones"]
    preset = ROOT / "data" / "videos" / "zones.json"  # saved zones for the demo clips
    if preset.exists():
        saved = json.loads(preset.read_text())
        if video.name in saved:
            return [dict(z, points=[tuple(p) for p in z["points"]]) for z in saved[video.name]]
    return [{"name": "Restricted area", "kind": "area", "points": [(0.6, 0.55), (0.98, 0.55), (0.98, 0.98), (0.6, 0.98)]}]


def rect_of(z: dict):
    xs, ys = [p[0] for p in z["points"]], [p[1] for p in z["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Geist, sans-serif", size=12, color="#9a9ca1"),
                      margin=dict(l=8, r=8, t=8, b=8), hoverlabel=dict(bgcolor="#1f2226", bordercolor="#333",
                                                                      font=dict(family="Geist Mono", color="#e8e6e1")))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)", zeroline=False, linecolor="rgba(255,255,255,.08)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", zeroline=False)
    return fig


def section(title: str, note: str = "") -> None:
    st.markdown(f'<div class="cl-h"><h3>{html.escape(title)}</h3><span>{html.escape(note)}</span></div>',
                unsafe_allow_html=True)


def draw_zones_ui(video: Path, key: str) -> None:
    """Drag a box (or lasso) on the frame, name it, add it as a zone."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        st.warning("Could not read a frame from this video.")
        return
    H0, W0 = f.shape[:2]
    img = cv2.cvtColor(cv2.resize(f, (960, int(H0 * 960 / W0))), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    fig = go.Figure(go.Image(z=img, hoverinfo="skip"))
    gx, gy = np.meshgrid(np.linspace(0, w, 64), np.linspace(0, h, 36))
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers", marker=dict(size=6, opacity=0.01),
                             hoverinfo="skip", showlegend=False))  # invisible grid so selections register
    for z in ss.zones:
        xs = [p[0] * w for p in z["points"]] + [z["points"][0][0] * w]
        ys = [p[1] * h for p in z["points"]] + [z["points"][0][1] * h]
        col, fill = ("#d46a5f", "rgba(212,106,95,.14)") if z["kind"] == "area" else ("#d6a250", "rgba(214,162,80,.14)")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=2), fill="toself", fillcolor=fill,
                                 hoverinfo="text", text=z["name"], showlegend=False))
        fig.add_annotation(x=min(xs) + 6, y=min(ys) + 6, text=z["name"], showarrow=False, xanchor="left", yanchor="top",
                           font=dict(color="#e8e6e1", size=12, family="Geist"), bgcolor=col, borderpad=3)
    style_fig(fig, 430)
    fig.update_layout(dragmode="select", margin=dict(l=0, r=0, t=34, b=0),
                      xaxis=dict(visible=False, range=[0, w], constrain="domain"),
                      yaxis=dict(visible=False, range=[h, 0], scaleanchor="x", constrain="domain"))
    ev = st.plotly_chart(fig, on_select="rerun", selection_mode=("box", "lasso"), key=f"zone_fig_{key}",
                         use_container_width=True, config={"displaylogo": False,
                                                           "modeBarButtonsToRemove": ["toImage", "autoScale2d"]})
    poly = None
    sel = getattr(ev, "selection", None) or (ev.get("selection") if isinstance(ev, dict) else None)
    if sel:
        lasso, box = sel.get("lasso") or [], sel.get("box") or []
        if lasso:
            poly = list(zip(lasso[-1]["x"], lasso[-1]["y"]))
        elif box:
            (x0, x1), (y0, y1) = sorted(box[-1]["x"]), sorted(box[-1]["y"])
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    c1, c2, c3 = st.columns([2.2, 1.3, 1.3], vertical_alignment="bottom")
    name = c1.text_input("Zone name", value=f"Zone {len(ss.zones) + 1}", key=f"nz_{key}_{len(ss.zones)}")
    kind = c2.selectbox("Type", ["area", "activity"], key=f"nk_{key}_{len(ss.zones)}",
                        help="area: logs entries and exits · activity: detects motion stopping, e.g. a machine")
    if c3.button("Add zone", disabled=poly is None, use_container_width=True, key=f"add_{key}", type="primary"):
        pts = [(round(min(max(x / w, 0), 1), 4), round(min(max(y / h, 0), 1), 4)) for x, y in poly]
        if len(pts) > 40:
            pts = pts[:: max(1, len(pts) // 40)]
        ss.zones.append({"name": name.strip() or f"Zone {len(ss.zones) + 1}", "kind": kind, "points": pts})
        ss.zone_rev = ss.get("zone_rev", 0) + 1
        st.rerun()
    st.caption("Drag a rectangle on the frame (or choose the lasso in the chart toolbar for a free shape), "
               "then add it. " + ("Selection ready." if poly else ""))


@st.cache_resource(show_spinner=False)
def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
    except Exception:  # noqa: BLE001
        return "CUDA"


def seek(t: float):
    ss.seek = max(0.0, float(t) - 1.0)


def timeline_fig(a, height: int | None = None, compact: bool = False) -> go.Figure:
    fig = go.Figure()
    ids = a.identities.set_index("id") if len(a.identities) else pd.DataFrame()
    order = sorted(a.segments, key=lambda g: ids.loc[g, "first_seen"] if g in ids.index else 0)
    colour = {g: ID_COLOURS[i % len(ID_COLOURS)] for i, g in enumerate(sorted(a.detections.gid.unique()))}
    label = {g: f"{g}  {ids.loc[g, 'cls'] if g in ids.index else ''}" for g in order}
    for g in order:
        for s, e in a.segments[g]:
            fig.add_trace(go.Bar(x=[max(e - s, 0.25)], y=[label[g]], base=[s], orientation="h", width=0.5,
                                 marker=dict(color=colour.get(g, "#888"), line=dict(width=0)), showlegend=False,
                                 hovertemplate=f"{g} visible {fmt_t(s)} – {fmt_t(e)}<extra></extra>"))
    ev = a.events
    for typ, sym, col, name in (("zone_enter", "triangle-right", "#e8e6e1", "zone entry"),
                                ("reappear", "diamond", "#d6a250", "re-identified")):
        sub = ev[(ev.type == typ) & ev.subject.isin(order)]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub.start, y=[label[s] for s in sub.subject], mode="markers", name=name,
                                     marker=dict(symbol=sym, size=10 if compact else 12, color=col,
                                                 line=dict(width=1, color="#0d0e10")),
                                     text=sub.details, hovertemplate="%{text} · %{x:.1f}s<extra></extra>"))
    for typ, col in (("activity_stop", "rgba(214,162,80,.16)"), ("sudden_change", "rgba(212,106,95,.18)"),
                     ("sound", "rgba(138,166,201,.16)")):
        for r in ev[ev.type == typ].itertuples():
            fig.add_vrect(x0=r.start, x1=max(r.end, r.start + 0.5), fillcolor=col, line_width=0, layer="below")
    if ss.get("qa") and ss.qa.timestamps:
        for t in ss.qa.timestamps[:6]:
            fig.add_vline(x=t, line=dict(color="#d6a250", width=1, dash="dot"))
    style_fig(fig, height or (120 + 34 * len(order)))
    fig.update_layout(barmode="overlay", showlegend=not compact, bargap=0.4,
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    fig.update_xaxes(range=[0, a.meta["duration"]], ticksuffix="s")
    return fig


# ----------------------------------------------------------------------------- sidebar
llm = LLM.from_env()
with st.sidebar:
    st.markdown('<div class="cl-side-brand"><span class="cl-mark"></span>ChronoLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="cl-label">Source</div>', unsafe_allow_html=True)
    clips = sorted(p for p in (ROOT / "data" / "videos").glob("*.mp4") if p != SAMPLE)
    src = st.radio("Source", ["Sample · warehouse", "Recorded clips", "Upload video"], label_visibility="collapsed")
    video = SAMPLE
    if src == "Recorded clips" and clips:
        video = st.selectbox("Clip", clips, format_func=lambda p: p.stem, label_visibility="collapsed")
    if src == "Upload video":
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv", "webm"], label_visibility="collapsed")
        if up:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            video = UPLOADS / f"{hashlib.sha1(up.getvalue()[:1 << 20]).hexdigest()[:10]}_{Path(up.name).name}"
            if not video.exists():
                video.write_bytes(up.getvalue())
    if ss.get("zones_for") != str(video):
        ss.zones = default_zones(video)
        ss.zones_for = str(video)
        ss.analysis, ss.qa = None, None

    st.markdown('<div class="cl-label">Zones</div>', unsafe_allow_html=True)
    rev = ss.get("zone_rev", 0)
    for i, z in enumerate(list(ss.zones)):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f'<div class="cl-zone"><span>{html.escape(z["name"])}</span><span class="k">{z["kind"]}</span></div>',
                    unsafe_allow_html=True)
        if c2.button("✕", key=f"zr{i}_{rev}", help=f"Remove {z['name']}"):
            ss.zones.pop(i)
            ss.zone_rev = rev + 1
            st.rerun()
    if not ss.zones:
        st.caption("No zones. Draw one on the frame.")
    with st.expander("Fine-tune zones"):
        for i, z in enumerate(ss.zones):
            z["name"] = st.text_input("Name", z["name"], key=f"zn{i}_{rev}")
            if len(z["points"]) == 4:
                x1, y1, x2, y2 = rect_of(z)
                xr = st.slider("Horizontal", 0.0, 1.0, (float(x1), float(x2)), 0.01, key=f"zx{i}_{rev}")
                yr = st.slider("Vertical", 0.0, 1.0, (float(y1), float(y2)), 0.01, key=f"zy{i}_{rev}")
                z["points"] = [(xr[0], yr[0]), (xr[1], yr[0]), (xr[1], yr[1]), (xr[0], yr[1])]
            else:
                st.caption(f"Free-form shape · {len(z['points'])} points")

    st.markdown('<div class="cl-label">Analysis</div>', unsafe_allow_html=True)
    with st.expander("Settings"):
        model = st.selectbox("Detector", ["yolo11s.pt", "yolo11n.pt", "yolo11m.pt"], help="n fastest · m most accurate")
        fps = st.slider("Samples per second", 2, 25, 10)
        stationary = st.slider("Stationary after (s)", 3, 180, 8)
        reid = st.slider("Re-identification threshold", 0.3, 0.9, 0.62, 0.01)
        captions = st.slider("Keyframe captions every N s (0 = off)", 0, 30, 0)
    settings = Settings(target_fps=fps, model=model, stationary_min=stationary, reid_threshold=reid,
                        captions_every=float(captions),
                        zones=[Zone(z["name"], [tuple(p) for p in z["points"]], z["kind"]) for z in ss.zones])
    run = st.button("Analyse video", type="primary", use_container_width=True, disabled=not video.exists())

    st.markdown('<div class="cl-label">Reasoning</div>', unsafe_allow_html=True)
    if llm.configured:
        st.markdown(f'<div class="cl-keyline"><span class="dot ok"></span>{html.escape(llm.model)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="cl-keyline"><span class="dot warn"></span>Offline rule engine (no API key)</div>',
                    unsafe_allow_html=True)

# ----------------------------------------------------------------------------- top bar
a = ss.analysis
chips = []
if a is not None:
    dev = a.meta.get("gpu") or (gpu_name() if str(a.meta.get("device", "")).startswith("cuda") else "CPU")
    chips.append(f'<span class="cl-chip"><span class="dot ok"></span><b>{html.escape(dev)}</b></span>')
    chips.append(f'<span class="cl-chip">{html.escape(a.meta.get("model", ""))} · BoT-SORT</span>')
    chips.append(f'<span class="cl-chip">processed in <b class="mono">{a.meta.get("processing_seconds", 0):.0f}s</b></span>')
else:
    chips.append('<span class="cl-chip"><span class="dot"></span>Not analysed</span>')
st.markdown(f'<div class="cl-top"><div><h1>{html.escape(video.name if video.exists() else "No video")}</h1>'
            f'<div class="sub">Timestamped answers about what happened, when and in what order.</div></div>'
            f'<div class="cl-chips">{"".join(chips)}</div></div>', unsafe_allow_html=True)

if not video.exists():
    st.info("Upload a video in the sidebar, or generate the sample with `python data/make_synthetic.py`.")
    st.stop()

if run:
    bar = st.progress(0.0, "Starting")
    try:
        ss.analysis = analyze(video, settings, progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), m), llm=llm)
        ss.qa = None
        ss.seek = 0.0
    except Exception as e:  # noqa: BLE001
        st.error(f"Analysis failed: {e}")
    bar.empty()
    st.rerun()

# ----------------------------------------------------------------------------- empty state
if a is None:
    c1, c2 = st.columns([1.7, 1], gap="large")
    with c1:
        section("Define zones", "drag on the frame")
        draw_zones_ui(video, "setup")
    with c2:
        section("Pipeline")
        st.markdown("""
<div class="cl-steps">
 <div class="cl-step"><div class="n">01</div><div><b>Detect and track</b><span>YOLO11 + BoT-SORT on the GPU, ten samples per second.</span></div></div>
 <div class="cl-step"><div class="n">02</div><div><b>Re-identify</b><span>Keeps one identity through occlusion and when people leave and return.</span></div></div>
 <div class="cl-step"><div class="n">03</div><div><b>Build the event log</b><span>Zone entries, stops, loitering, interactions, alarms and sounds, each with an exact time.</span></div></div>
 <div class="cl-step"><div class="n">04</div><div><b>Answer with evidence</b><span>Queries run over the event log and are cross-checked. Unsupported claims are reported as not observed.</span></div></div>
</div>""", unsafe_allow_html=True)
        st.write("")
        st.caption("Set zones, then run **Analyse video** from the sidebar.")
    st.stop()

# ----------------------------------------------------------------------------- results
m = a.meta
ids = a.identities
n_people = int((ids.kind == "person").sum()) if len(ids) else 0
n_other = int((ids.kind != "person").sum()) if len(ids) else 0
stats = [("Duration", fmt_t(m["duration"]), ""), ("People", n_people, ""), ("Vehicles · objects", n_other, ""),
         ("Events", len(a.events), ""), ("Re-ID links", len(a.links), ""),
         ("Sampling", f"{m['sample_fps']:.1f}", "/s")]
st.markdown('<div class="cl-stats">' + "".join(
    f'<div class="cl-stat"><div class="l">{lbl}</div><div class="v">{val}<small>{unit}</small></div></div>'
    for lbl, val, unit in stats) + "</div>", unsafe_allow_html=True)

tab_ask, tab_tl, tab_ids, tab_ev, tab_zones, tab_bench = st.tabs(
    ["Investigate", "Timeline", "Identities", "Event log", "Zones", "Benchmark"])

with tab_ask:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        section("Annotated video", f"from {fmt_t(ss.seek)}")
        st.video(str(a.annotated), start_time=int(ss.seek), autoplay=ss.seek > 0, muted=True)
        st.plotly_chart(timeline_fig(a, height=60 + 26 * max(1, len(a.segments)), compact=True),
                        use_container_width=True, config={"displayModeBar": False}, key="mini_tl")
    with right:
        section("Ask", "answers cite events and timestamps")
        pick = st.pills("Suggestions", list(SUGGESTIONS), selection_mode="single", label_visibility="collapsed",
                        key="sugg")
        if pick and ss.get("last_pick") != pick:
            ss.question, ss.last_pick = SUGGESTIONS[pick], pick
        with st.form("ask", border=True):
            q = st.text_area("Question", value=ss.get("question", ""), height=80, label_visibility="collapsed",
                             placeholder="What happened right before the alarm?")
            c1, c2 = st.columns([1.6, 1], vertical_alignment="center")
            mode = c1.segmented_control("Mode", ["Cross-checked", "Offline"], default="Cross-checked" if llm.configured
                                        else "Offline", label_visibility="collapsed")
            go_ = c2.form_submit_button("Ask", type="primary", use_container_width=True)
        if go_ and q.strip():
            ss.question = q
            with st.spinner("Querying the event log"):
                ss.qa = ask(a, q.strip(), llm, mode="offline" if mode == "Offline" else "auto")
            if ss.qa.timestamps:
                seek(ss.qa.timestamps[0])
            st.rerun()

        r = ss.qa
        if r is not None:
            label = {"answered": "Answer", "not_observed": "Not observed", "error": "Could not answer"}[r.status]
            conf = int(r.confidence * 100)
            prov = r.engine or "—"
            if r.attempts > 1:
                prov += f" · self-corrected {r.attempts - 1}×"
            check = {"agrees": "Agrees with rule engine",
                     "overrode LLM": '<span class="cl-warn">LLM disagreed with the evidence; rule engine answer shown</span>'
                     }.get(r.crosscheck, "Not applicable")
            meta_rows = ""
            if r.status != "error":
                meta_rows = (f'<div class="k">Confidence</div><div><span class="mono">{conf}%</span>'
                             f'<span class="cl-bar"><i style="width:{conf}%"></i></span></div>'
                             f'<div class="k">Engine</div><div>{html.escape(prov)}</div>'
                             f'<div class="k">Cross-check</div><div>{check}</div>')
            st.markdown(f'<div class="cl-answer {r.status}"><div class="cl-status">{label}</div>'
                        f'<p class="a">{html.escape(r.answer or r.error)}</p>'
                        f'<div class="cl-meta">{meta_rows}</div></div>', unsafe_allow_html=True)
            if r.timestamps:
                st.write("")
                tc = st.columns(min(4, len(r.timestamps)))
                for i, t in enumerate(r.timestamps[:4]):
                    if tc[i].button(fmt_t(t), key=f"ts{i}", use_container_width=True, help="Play from here"):
                        seek(t)
                        st.rerun()
            if r.evidence:
                st.write("")
                rows = "".join(
                    f'<div class="cl-ev-row"><span class="id">{e["id"]}</span><span class="t">{html.escape(str(e["when"]))}</span>'
                    f'<span class="c">{e["confidence"]:.2f}</span><span class="d">{html.escape(str(e["details"]))}</span></div>'
                    for e in r.evidence[:10])
                st.markdown(f'<div class="cl-box">{rows}</div>', unsafe_allow_html=True)
            with st.expander("Query code"):
                st.code(r.code or "# answered by the deterministic rule engine", language="python")

    if ss.qa is not None and ss.qa.timestamps:
        st.write("")
        section("Evidence", "frames with the involved identities highlighted · clips ±3 s")
        view = st.segmented_control("Evidence view", ["Frames", "Clips"], default="Frames", label_visibility="collapsed",
                                    key="ev_view")
        ts = ss.qa.timestamps[:4]
        cols = st.columns(len(ts))
        for i, (t, c) in enumerate(zip(ts, cols)):
            with c:
                if view == "Clips":
                    clip = export_clip(a, t)
                    if clip:
                        st.video(str(clip))
                        st.download_button(f"Download {fmt_t(t)}", clip.read_bytes(), file_name=clip.name,
                                           mime="video/mp4", key=f"dl{i}", use_container_width=True)
                else:
                    near = a.detections[(a.detections.t - t).abs() <= 0.6 / max(m["sample_fps"], 1) + 0.05]
                    if ss.qa.subjects:
                        near = near[near.gid.isin(ss.qa.subjects)]
                    img = frame_at(a.video, t, 520, near.sort_values("t").drop_duplicates("gid"), m)
                    if img is not None:
                        st.image(img, use_container_width=True)
                st.markdown(f'<div class="cl-id"><b>{fmt_t(t)}</b></div>', unsafe_allow_html=True)

with tab_tl:
    section("Activity timeline", "bars: identity visible · ▸ zone entry · ◆ re-identified · shaded: stops, alarms, sounds")
    st.plotly_chart(timeline_fig(a), use_container_width=True, config={"displaylogo": False}, key="full_tl")

with tab_ids:
    if ids.empty:
        st.info("No identities were detected.")
    else:
        section("Identities", f"{len(ids)} tracked")
        import base64

        cards = []
        for i, row in enumerate(ids.itertuples()):
            p = a.thumb(row.id)
            img = (f'<img src="data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{row.id}">'
                   if p else "")
            col = ID_COLOURS[sorted(a.detections.gid.unique()).index(row.id) % len(ID_COLOURS)]                 if row.id in set(a.detections.gid) else "#888"
            reid = (f'<div class="r"><span>Re-ID</span><span class="mono">{row.reid_links} link'
                    f'{"s" if row.reid_links != 1 else ""} · {row.reid_score:.2f}</span></div>' if row.reid_links else "")
            cards.append(
                f'<div class="cl-card-id"><div class="th">{img}</div><div class="hd"><i style="background:{col}"></i>'
                f'<b>{row.id}</b><span>{html.escape(str(row.cls))}</span></div>'
                f'<div class="r"><span>Seen</span><span class="mono">{row.first} – {row.last}</span></div>'
                f'<div class="r"><span>Visible</span><span class="mono">{row.visible_seconds:.0f}s</span></div>{reid}</div>')
        st.markdown('<div class="cl-id-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        if len(a.links):
            st.write("")
            section("Re-identification decisions", "why tracklets were linked")
            st.dataframe(a.links[["identity", "from_track", "to_track", "score", "gap", "reason"]], hide_index=True,
                         use_container_width=True)

with tab_ev:
    types = sorted(a.events.type.unique())
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    pick_t = c1.multiselect("Event types", types, default=[t for t in types if t != "in_zone"])
    c2.download_button("Export CSV", a.events.to_csv(index=False), "events.csv", "text/csv", use_container_width=True)
    view = a.events[a.events.type.isin(pick_t)][["id", "when", "type", "subject", "zone", "other", "details", "confidence"]]
    sel = st.dataframe(view, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row",
                       height=520, column_config={"confidence": st.column_config.ProgressColumn(
                           "confidence", min_value=0, max_value=1, format="%.2f")})
    if sel and sel.selection.rows:
        row = view.iloc[sel.selection.rows[0]]
        t0 = float(a.events.set_index("id").loc[row.id, "start"])
        st.button(f"Play from {fmt_t(t0)} in Investigate", on_click=seek, args=(t0,))

with tab_zones:
    section("Zones", "area zones update instantly on re-analysis; activity zones re-run tracking")
    draw_zones_ui(video, "results")

with tab_bench:
    rep = ROOT / "eval" / "REPORT.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.info("Run `python eval/run_eval.py --auto` to score ChronoLens on the ground-truth benchmark.")
