# ChronoLens — demo script & judge Q&A

## Before you go on stage (10 min earlier)

1. Start the app: `streamlit run app.py` (in WSL: `~/cv312/bin/streamlit run app.py`). Open http://localhost:8502.
2. Click **Analyse video** on the sample. It's cached, so it loads instantly.
3. Ask every demo question once, so the answers are cached and replay instantly even if Wi-Fi dies.
4. Have your own phone clip already analysed too (pick **Upload a video**).
5. Keep a terminal ready with `python eval/run_eval.py --offline`. It proves 13/13 in about 20 s with no internet.

## 3-minute demo

**0:00 — The problem (20 s)**
> "Video Q&A is scored on *when* things happened, not just *what*. Vision-language models describe frames well but
> guess timestamps. So we never let the AI guess a time."

**0:20 — Architecture (30 s)** (show the README diagram or say it)
> "The GPU detects and tracks every person and object 10 times a second. Re-identification keeps the same ID when
> someone is hidden or leaves and comes back. A deterministic event engine turns that into an exact, timestamped
> event log. The LLM only writes a query over that log, so every time in an answer comes from a detected event."

**0:50 — Live: Ask tab (60 s)**
1. *"What happened right before the safety alarm went off?"* → P2 entered the restricted area at 1:06, 4 s before
   the alarm at 1:10. Click **▶ 1:06**: the video jumps there and the evidence frame shows P2 inside the red zone.
2. *"How many times did the machine stop?"* → 2, with both intervals.
3. *"Did the first person come back later?"* → yes, at 0:41. Mention that the person left the frame for 23 s, so the
   tracker gave them a new ID, and re-identification linked them back (show the **Identities** tab with the link
   score and reason).
4. *"When did a dog run across the floor?"* → **Not observed**. "It refuses instead of hallucinating."

**1:50 — Timeline tab (15 s)**
> "Every identity's visibility, zone entries, re-identifications, machine stops and the alarm on one axis."

**2:05 — Trust (40 s)**
> "Every answer shows its supporting event IDs, a confidence score, and a cross-check: a deterministic rule engine
> answers the same question independently. If the LLM disagrees with the evidence, we show the evidence. We scored
> it on a video with exact ground truth using your judging rule: **13 out of 13**. And with the internet switched
> off, the rule engine alone also scores **13/13**."
(Optional: run `python eval/run_eval.py --offline` live.)

**2:45 — Real footage + close (15 s)**
> "Here's a clip we filmed this morning." Ask one question about it.
> "Accurate timestamps, persistent identities, evidence for every answer, and it says 'not observed' when it
> doesn't know."

## Likely judge questions and short answers

**Q: How are your timestamps accurate?**
Detection runs about 10 times per second, so resolution is about 0.1 s. Zone entry uses the person's *feet* point
with hysteresis, so standing on the boundary doesn't flicker. The LLM never produces a time; it filters events
whose times were measured. On the benchmark every time was within 0.4 s of ground truth.

**Q: How do you know it's the same person after they leave or get hidden?**
The tracker keeps an ID through short occlusions (6 s buffer). For longer gaps we link "tracklets" using appearance
(CNN embedding + clothing colour histograms of upper and lower body) plus physical plausibility: time gap, where
they left versus where they reappeared, frame edges. Two tracks that exist at the same time can never be the same
person. Each link has a score that appears in the UI, and answers that depend on it inherit that confidence.

**Q: What if two people look alike?**
Our benchmark includes exactly that: P3 is a mirrored copy of P2's body. They're kept apart because they are on
screen at the same time, and their colour signatures differ. Honest limitation: two *identical-looking* people who
are never on screen together could be merged. A dedicated person re-ID model would help; that's our next step.

**Q: Why not just use GPT-4V / Gemini on the video?**
They give vague or invented timestamps and can't count reliably over long videos. We use them only optionally, to
caption keyframes for things a detector can't name, and those captions carry exact keyframe times.

**Q: How does "right before X" work?**
We find X's time in the event log (e.g. the alarm sound at 1:10.0), then take the latest relevant event that
starts before it, within 30 s. The answer cites both events.

**Q: How did you detect the alarm?**
Two independent signals: the audio track (a narrow-band tone rising 29 dB above background, detected
with an FFT) and the video (a red-light / brightness spike). Both agree on 1:10.0.

**Q: How is the machine stop detected?**
You draw an "activity zone" over the machine. We measure pixel motion energy inside it every sample, auto-calibrate
the running vs stopped levels, and drop blips shorter than 1.5 s. Stops are intervals with start and end times.

**Q: What happens with a long video?**
The sampling rate drops automatically (6/s above 5 min). A 10-minute video takes about 2 min 15 s on our RTX 5070
laptop. The analysis is cached, so questions afterwards are instant.

**Q: What about a moving camera?**
BoT-SORT includes camera-motion compensation, so tracking still works. Zones are fixed in image coordinates, so
zone events assume a fixed camera; we say that in the scope note.

**Q: What if the LLM is wrong?**
Three safeguards: (1) its code only reads the event log, so it can't invent events; (2) cited event IDs are
validated; (3) a deterministic rule engine answers the same question, and if the two disagree we show the
evidence-backed answer. In testing, that cross-check caught a weak model saying "0 stops" and replaced it with 2.

**Q: Is the benchmark fair? You made the video.**
Yes, that's why it has exact ground truth: every event time is scripted. It's built from real photographs of
people, includes occlusion, leave-and-return and a look-alike, and is scored with your rule. We also show real
phone footage that we didn't script frame by frame.

**Q: What didn't you finish?**
Cross-camera re-ID, a dedicated person re-ID network, zone tracking for moving cameras, and action recognition beyond
what detection and motion give us (that's covered optionally by VLM captions).

**Q: Which parts did you build vs. use?**
Used: YOLO11 + BoT-SORT (ultralytics), ResNet-18 weights, Gemini/Claude API, Streamlit. Built: the
re-identification logic, the event engine (11 event types + audio), the temporal QA layer with validation and
cross-check, the offline rule engine, the benchmark generator and scorer, and the UI.

## If something breaks live

| Problem | Do this |
|---|---|
| Wi-Fi / API down or "quota" error | Switch the answer mode to **Offline rules**. Same answers for the demo questions. |
| Analysis is slow on a judge's video | Analysis settings → Detector `yolo11n.pt`, frame rate 5 |
| Wrong zone | Zones panel → adjust the x/y sliders → Analyse again (tracking is cached, only events recompute) |
| App crashes | Restart `streamlit run app.py`; analyses and answers are cached on disk |
