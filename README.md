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
In the app, pick the sample or upload any video, draw zones by dragging on the frame (area = restricted area,
activity = machine),
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

### Result (13 questions, judging rule, ±1.5 s)

| answering mode | score | notes |
|---|---|---|
| **hybrid (default): LLM + deterministic cross-check** | **13 / 13** | the rule engine caught a wrong "0 times" from a weak fallback model and replaced it with the evidence |
| **offline rule engine (no internet, no LLM)** | **13 / 13** | answers counts, times, before/after, ordering, loitering, returns, sounds and time-in-zone from the event log |
| LLM only (Gemini 3.8 Flash) | 12 / 12 | first benchmark version without the audio question |
| LLM only (free-tier fallback "lite" models) | 11 / 13 | why the cross-check exists |

The 100-second benchmark video is analysed in about 30 s on an RTX 5070 Laptop GPU; a **10-minute video in 2 min 15 s**
(the frame rate is lowered automatically for long clips). Perception alone, before any LLM, recovers the scripted truth:

| ground truth | ChronoLens event log |
|---|---|
| machine stops at 20.0 s and 60.0 s | 0:20.1 and 1:00.1 |
| alarm (beeping tone + flashing light) at 70.0 s | 1:10.0, from both audio and video |
| crash sound at 35.0 s | 0:35.0 (audio) |
| P2 leaves zone at 46.44 s / 74.84 s | 0:46.4 / 1:14.9 |
| loiterer still 58–92 s | 0:57.6–1:33.4 |
| 3 people (one leaves & returns, one crosses behind a pillar 3×, one is a mirrored look-alike) | 3 identities, all re-ID links correct |

Full tables: `eval/REPORT.md` (hybrid), `eval/REPORT_offline.md`, `eval/REPORT_llm_only.md`.

### Real-world validation: hand-held phone footage

Three ~1-minute clips filmed in a college corridor on a hand-held phone (478×850 portrait, compressed, people
about 40–260 px tall, a 30 px bag on the floor, the phone moving and even zooming). The recordings show real
people, so they are kept out of this public repository; their saved zones are in `data/videos/zones.json` and the
answers in `eval/real_clips_results.json`. Cross-checked mode, all answers verified against the footage:

| clip | question | ChronoLens | ✓ |
|---|---|---|---|
| bag | Was any bag left behind? Who left it, how long untouched? | B3 left by P1 at 0:15.0, untouched 28.8 s | ✅ |
| bag | Did the person who left the bag come back for it? | yes, at 0:42.4 (re-identified after 26 s, VLM-confirmed 0.95) | ✅ |
| bag | How many different people appear? | 1 | ✅ |
| door | Who stood at the restricted door, and for how long? | P2 from 0:17.2, 8.4 s | ✅ |
| door | Did the first person stop at the restricted door? | no (P1 walked past) | ✅ |
| door | How many different people appear? | 2 | ✅ |
| corridor | Who stood still for more than 20 seconds? | P2, 0:17.4–0:51.9 | ✅ |
| corridor | Did the first person come back later? | yes, at 0:59.1 (after 49 s away) | ✅ |
| corridor | When did a dog run across the corridor? | not observed | ✅ |

All nine LLM answers agreed with the deterministic rule engine. What it took to get there (and is now general):
camera registration for hand-held footage, persistence-validated left-object detection, a brightness-aware clothing
signature, and vision-model verification of grey-zone re-identifications.

### Reliability features
* **Camera registration:** every frame is registered to the zone-drawing frame (ORB features + RANSAC similarity
  transform: translation, zoom, rotation). Zones, stillness and left objects are evaluated in scene coordinates;
  camera-induced motion and exposure jumps are not reported as events. A fixed camera measures as exactly static.
* **Re-identification you can audit:** appearance (CNN + torso/legs colour-and-brightness signature) plus physical
  plausibility; a "clearly best" rule only applies when there is a real competitor; grey-zone pairs are checked by a
  vision-language model on the two best snapshots (max 8 checks per video). Doubtful cases stay separate.
* **Left-behind objects:** a sensitive bag pass twice per second, kept only if a detection persists in the same scene
  position for 5+ s; "left by" = nearest person when it appeared; items inside a person's box count as held.
* **Cross-check:** when the deterministic rule engine also understands a question, its answer is compared with the
  LLM's (values and key timestamps). If they disagree, the evidence-backed rule answer is shown and the disagreement
  is reported. "Not observed" from the LLM is overridden when the rule engine finds supporting events.
* **Offline mode:** no internet or no quota still gives exact answers for common questions.
* **Model fallback:** HTTP 503/429 (overload / free-tier daily quota) switches across several models automatically,
  and responses are cached, so a live demo never stalls.
* **Audio events:** loud sounds and alarm-like tones (sirens, beepers) from the soundtrack, with an adaptive threshold.
* **Ghost filters:** short low-confidence tracks, duplicate boxes on one person, faint "people" inside vehicles and
  held objects are removed.

## Business model

**The problem we sell against:** when something happens on camera, someone has to scrub through hours of CCTV
to find it. ChronoLens turns footage into a searchable, timestamped timeline of events, so an investigator asks
*"Who entered the restricted area?"*, *"What happened before the alarm?"* or *"How many times did this happen?"*
and gets the exact moment, the people or objects involved, and the evidence, in seconds.

**Customers:** factories and manufacturing plants, warehouses and logistics, construction sites, retail stores,
schools and colleges, and security service companies.

| revenue stream | what the customer gets | status in this prototype |
|---|---|---|
| **SaaS subscription** (monthly / annual, priced per camera and hours analysed) | upload or connect footage, ask questions, timeline, identities, event log | working (single camera, uploaded video) |
| **Incident reports** (included in higher tiers, or per report) | one-click report: key findings, every question with timestamps and evidence frames, full event log | **working:** *Report* tab, printable to PDF |
| **Real-time alerts** (add-on) | notifications for restricted-area entry, left-behind objects, loitering, machine stops, alarms | event engine exists; live-stream input and notifications are roadmap |
| **Enterprise** | many cameras, API integration, on-premise / private deployment (footage never leaves the site), custom detectors | runs fully on a local GPU today; API and multi-camera are roadmap |

**Why customers would trust it:** every answer cites timestamped events and evidence, a rule engine cross-checks the
AI, and the system says *"not observed"* instead of guessing. An incident report is only useful if it holds up.

**Positioning:** we don't just record what happened. We understand what happened, when it happened, and what happened
before and after.

## Scope note

**Implemented (MVP + advanced):** one-click incident reports (findings, Q&A with evidence frames, event log) ·
GPU detection and tracking · re-identification across occlusion and re-entry with
vision-model verification · camera registration for hand-held footage · left-object detection with attribution ·
deterministic event engine (12 event types + audio) · click-to-draw area/activity zones · LLM temporal QA with
executed, validated queries, cited evidence and a rule-engine cross-check · offline rule engine · "not observed"
refusals · per-answer evidence frames and clips · annotated video · timeline · identity gallery with re-ID decisions ·
event log with seek · ground-truth benchmark + scorer · real-footage validation · optional VLM keyframe captions.

**Limitations:** single camera per analysis (no cross-camera re-ID); identity for very small, distant people leans on
the vision-model check; actions beyond movement, zones, objects and sounds need the optional VLM captions; a camera
that travels far from the reference view loses zone registration; a dedicated person re-ID network would be stronger
than generic ImageNet features.

## Technologies, models and resources (declared)
* **Ultralytics YOLO11** (detection; segmentation only to build the synthetic benchmark), BoT-SORT tracker
  (bundled with ultralytics), **PyTorch / torchvision ResNet-18** (ImageNet weights) for appearance embeddings.
* OpenCV (incl. ORB features + RANSAC for camera registration), NumPy, pandas, imageio-ffmpeg (H.264), Streamlit, Plotly.
* LLM for question answering and re-ID verification: Google Gemini (used in testing) or Anthropic Claude / any
  OpenAI-compatible model, via a dependency-free client.
* Sample images `bus.jpg` from the ultralytics package (AGPL-3.0 assets) are used to build the synthetic video.
* AI coding assistants were used during development. The team reviewed and understands the code.

## Layout
```
app.py                       Streamlit UI (Investigate · Timeline · Identities · Event log · Zones · Benchmark)
chronolens/camera.py         camera registration for hand-held footage (ORB + RANSAC)
chronolens/audio.py          audio events (loud sounds, alarm-like tones)
chronolens/offline.py        deterministic rule engine (offline answers + cross-check)
chronolens/report.py         incident report generator (printable HTML)
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
