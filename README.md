# ChronoLens — video understanding with timestamp-exact temporal reasoning

**HackNex 2026 · Internal Qualifier · HNX26PSI02 — Video Understanding & Temporal Reasoning**

> Ask a video what happened, when, in which order, how often and for how long. Every answer carries
> timestamps, the IDs of the people and objects involved, the supporting events and evidence frames, or
> it honestly says *"not observed in the video"*.

## The idea

Vision-language models are good at describing a frame and bad at time: they give vague answers and
made-up timestamps, which this problem scores at zero or half credit. ChronoLens splits the job:

| layer | technology | gives |
|---|---|---|
| **Perception** (GPU) | YOLO11 detection + BoT-SORT tracking, 10 samples/s | boxes + tracker IDs for people, vehicles, bags and objects |
| **Re-identification** | ResNet-18 embeddings + clothing colour histograms + motion plausibility | one identity across occlusion and leaving/returning, with a match score |
| **Event engine** (deterministic) | geometry + signal processing | exact, timestamped events: zone enter/exit, stationary, untouched objects, interactions, machine stops, alarms |
| **Reasoning** | LLM writes a pandas query over the event log; we execute it | answers whose times come from detected events, never from the model |

## Architecture

```
video ──► [Perception: YOLO11 + BoT-SORT on GPU] ── boxes, track IDs, appearance features, motion signals
                     │
                     ▼
          [Re-identification] ── links tracklets: appearance (CNN + colour) + hand-over plausibility
                     │            (time gap, exit/entry position, frame edges, "re-appeared where lost")
                     ▼
          [Event engine] ── enter/exit view · occluded/reappear · zone enter/exit (feet point, hysteresis)
                     │       stationary · untouched · person↔object interaction · activity stop/start
                     │       (motion energy in a zone, e.g. a machine) · sudden change (motion/brightness/red light)
                     │       (+ optional VLM keyframe captions as extra evidence)
                     ▼
question ──► [LLM planner] ── writes answer(events, ids, tracks, series, meta) ── AST-checked, sandboxed exec
                     ▼
answer + timestamps + event IDs + subjects + evidence frames + confidence ──► UI (click a time → video seeks)
```

### Why timestamps are right
* Times come only from the event log, which is computed from per-sample detections (10 per second by default,
  so ±0.1 s resolution).
* Zone entry uses the **feet point** (bottom-centre of the box) with hysteresis, so a person standing at the
  edge doesn't generate dozens of enter/exit events.
* Machine stops come from motion energy inside an activity zone, with auto-calibrated thresholds and
  minimum-duration de-noising.
* The LLM must cite event IDs. Cited IDs are validated against the log, and the confidence drops if the reported
  times don't match any cited event.

### Identity across occlusion and re-entry
The tracker keeps an ID through short occlusions (6 s buffer). For longer gaps, the re-identification module
links tracklets that never overlap in time, are the same kind, look alike, and have a physically plausible
hand-over. Every link is shown with its score and reason in the UI, and answers that depend on a link inherit its
score as their confidence.

## Install & run

Requires Python 3.10+, ideally an NVIDIA GPU (CPU works, just slower).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128
pip install -r requirements.txt
cp .env.example .env          # add ONE LLM key (Claude / Gemini / Groq / any OpenAI-compatible)
python data/make_synthetic.py # renders the benchmark video + ground truth
streamlit run app.py          # http://localhost:8502
```
In the app, pick the sample or upload any video, adjust zones (area = restricted area, activity = machine),
click **Analyse video**, then ask questions. Every timestamp in an answer is a button that seeks the video.

### Reproduce the benchmark
```bash
python tests/test_engine.py   # re-ID + event engine unit tests (no GPU)
python eval/run_eval.py       # analyse the benchmark video, answer 12 questions, score -> eval/REPORT.md
```

## Benchmark video with exact ground truth (`data/make_synthetic.py`)

Real people and a real vehicle are cut out (YOLO11 segmentation) of the sample photos shipped with ultralytics,
recoloured into distinct identities, and animated over a workshop scene with exactly scripted events:
a restricted area, a foreground pillar that hides people, a person who leaves and returns, a delivery vehicle
that arrives/parks/leaves, a machine whose fan stops twice, a flashing alarm, and a person loitering.
Twelve questions cover ordering, cause and effect, counting, duration, re-identification, temporal overlap and an
unanswerable trap. They are scored with the problem statement's rule: right answer and right time (±1.5 s) = 1,
right answer with the wrong time = 0.5.

See `eval/REPORT.md` for the latest score.

## Scope note

**Implemented (MVP + advanced):** GPU detection and tracking · re-identification across occlusion and re-entry ·
deterministic event engine (11 event types) · configurable area/activity zones · LLM temporal QA with
executed, validated queries and cited evidence · "not observed" refusals · annotated video export · interactive
timeline · identity gallery with re-ID decisions · event log with seek · ground-truth benchmark + scorer ·
optional VLM keyframe captions.

**Stretch / limitations:** single camera per analysis (no cross-camera re-ID); moving-camera footage is handled by
BoT-SORT's camera-motion compensation for tracking, but zones are fixed in image coordinates; actions outside COCO
classes rely on optional VLM captions; re-ID uses generic ImageNet features (a dedicated person re-ID model would be
stronger); audio events are not analysed.

## Technologies, models and resources (declared)
* **Ultralytics YOLO11** (detection; segmentation only to build the synthetic benchmark), BoT-SORT tracker
  (bundled with ultralytics), **PyTorch / torchvision ResNet-18** (ImageNet weights) for appearance embeddings.
* OpenCV, NumPy, pandas, imageio-ffmpeg (H.264 encoding), Streamlit, Plotly.
* LLM for question answering: Anthropic Claude (default) or any OpenAI-compatible model, via a dependency-free client.
* Sample images `bus.jpg` from the ultralytics package (AGPL-3.0 assets) are used to build the synthetic video.
* AI coding assistants were used during development. The team reviewed and understands the code.

## Layout
```
app.py                       Streamlit UI (Ask · Timeline · Identities · Event log · Benchmark)
chronolens/perception.py     YOLO11 + BoT-SORT pass, appearance features, motion signals
chronolens/reid.py           tracklet linking into identities
chronolens/events.py         deterministic event engine
chronolens/qa.py             LLM planner + sandboxed execution + confidence
chronolens/pipeline.py       orchestration, caching, annotated video rendering
chronolens/captions.py       optional VLM keyframe captions
data/make_synthetic.py       benchmark video + ground truth generator
eval/run_eval.py             scorer (judging rule: right answer + right time)
tests/test_engine.py         unit tests for re-ID + events
```
