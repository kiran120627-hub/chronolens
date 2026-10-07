# ChronoLens — demo script & judge Q&A

## Before you go on stage (10 minutes earlier)

1. Start the app (Windows terminal):
   `wsl -e bash -lc "cd /mnt/c/Users/KIRAN/.claude/calde/chronolens && ~/cv312/bin/streamlit run app.py"`
   and open http://localhost:8502.
2. Sidebar → **Recorded clips** → `clip1` → **Analyse video**. Repeat for `clip2`, `clip3` and the **Sample**.
   Everything is cached, so each loads in seconds.
3. Ask each demo question below once. The answers are cached and replay instantly, even without Wi-Fi.
4. Keep a second terminal ready with `python eval/run_eval.py --offline`. It proves 13/13 in a few seconds, offline.
5. Laptop on charger, display at 100 % scale, browser full screen (F11).

## 3-minute demo

**0:00 — The problem (20 s)**
> "This problem is scored on *when* things happened, not just *what*. A wrong timestamp gets half credit.
> Vision-language models describe frames well but guess times. So in ChronoLens the AI never produces a timestamp."

**0:20 — How it works (30 s)**
> "The GPU detects and tracks every person and bag ten times a second. Re-identification keeps one identity when
> someone is hidden or leaves and comes back. A deterministic event engine turns that into an exact, timestamped
> event log. The LLM only writes a query over that log, and a rule engine cross-checks its answer."

**0:50 — Our own corridor footage, filmed on a hand-held phone (75 s)**
- **clip1**, ask *"Was any bag left behind? Who left it, and how long was it untouched?"*
  → *"B3 left by P1 at 0:15.0, untouched for 28.8 s."* Click the timestamp: the video jumps there.
  Then *"Did the person who left the bag come back for it?"* → *"Yes, at 0:42.4."*
  Open **Identities**: "He left the frame for 26 s and came back facing the other way, carrying nothing. Colour alone
  was unsure, so the system asked a vision model, which confirmed 'same person, 0.95'."
- **clip2**, ask *"Who stood at the restricted door, and for how long?"* → *"P2, from 0:17.2, for 8.4 s."*
  Then *"Did the first person stop at the restricted door?"* → *"No."*
  "The phone moved and even zoomed in, but the zone stays attached to the door, because every frame is registered
  to the reference frame."
- **clip3**, ask *"Who stood still for more than 20 seconds?"* → *"P2 from 0:17.4 to 0:51.9."*
  Then *"When did a dog run across the corridor?"* → **Not observed**. "It refuses instead of hallucinating."

**1:55 — Learned rhythm (15 s)** (Recorded clips → `machine_rhythm` → *Unexpected stops*)
> "The booklet asks how many times a machine stopped *unexpectedly*. We don't hard-code 'expected': the system
> learns the machine's rhythm from the video (every 30 s for 4 s) and flags the 2 stops that broke it: one came
> 9 seconds early, one lasted 10 seconds."

**2:05 — Proof (30 s)** (Sample video → **Benchmark** tab)
> "To measure accuracy we built a video with exactly scripted events: occlusion, people leaving and returning, a
> look-alike, a machine that stops twice, an alarm. Scored with your judging rule, ChronoLens gets **13 out of 13**.
> With the internet off, the rule engine alone also gets **13/13**."

**2:30 — The product (20 s)** (**Report** tab → **Generate report**)
> "Who pays for this? Factories, warehouses, construction sites, retail, colleges and security companies. Today a
> person spends hours scrubbing CCTV after an incident. We turn the footage into a searchable timeline. We'd sell it
> as a subscription per camera, with alerts and enterprise plans on top. And here's a paid feature that already
> works: one click gives a complete incident report, with findings, every question, timestamps and evidence frames."

**2:50 — Trust + close (20 s)**
> "Every answer cites event IDs, timestamps, an evidence frame and a 6-second clip, plus a confidence score and
> whether the rule engine agreed. Accurate times, persistent identities, evidence for every claim, and an honest
> 'not observed'."

## Likely judge questions and short answers

**Q: How are your timestamps accurate?**
Detection runs about 10 times per second, so resolution is about 0.1 s. Zone entry uses the person's feet point with
hysteresis. The LLM never produces a time; it filters events whose times were measured. On the benchmark every
time was within 0.4 s of ground truth.

**Q: How do you know it's the same person after they leave or get hidden?**
The tracker keeps an ID through short occlusions. For longer gaps we link tracks by appearance (CNN embedding plus a
clothing signature of torso and legs: colour *and* brightness) and physical plausibility (time gap, exit and entry
position). Two tracks that exist at the same time can never be the same person. If the score is in a grey zone,
a vision-language model compares the two best snapshots, ignoring pose and carried bags. Every link shows its score
and reason in the Identities tab.

**Q: What if two people look alike?**
The benchmark includes a look-alike (a mirrored, recoloured copy of another person), and our corridor clips have two
people. Neither was merged. Merging only happens when someone is clearly the best match, so a doubtful case stays
separate. Missing a return is safer than merging two people.

**Q: The phone was hand-held. How do zones still work?**
Every frame is registered to the reference frame with ORB feature matching and RANSAC (translation, zoom, rotation).
Moving people are rejected as outliers, so a fixed camera measures as "no motion". Zones, stillness and left objects
are all evaluated in scene coordinates, and camera-caused motion or exposure jumps are not reported as events.

**Q: The bag is tiny. How did you find it?**
A normal detector was only 4–10 % confident about it. Twice a second we run a sensitive pass for bags and keep only
detections that stay in the same scene position for 5+ seconds, because noise doesn't stay still. "Left by" is the
nearest person when it appeared; things inside a person's box (a phone, a bag in hand) are treated as held, not left.

**Q: Why not just use a VLM on the whole video?**
It gives vague or invented timestamps and can't count reliably over long videos. We use the VLM only where it's
strongest: comparing two snapshots for re-identification, or optional keyframe captions with exact times.

**Q: How does "right before X" work?**
Find X's time in the event log, then the latest relevant event before it (within 30 s). The answer cites both.

**Q: How did you detect the alarm?**
In the benchmark, two independent signals: the audio (a narrow-band tone 29 dB above background, found with an FFT)
and the video (a red-light / brightness spike). Both agree on 1:10.0.

**Q: How do you know a stop was "unexpected"?**
We learn the normal cycle from the video: the typical gap between stops and the typical length, fitted as a time
grid so one early stop doesn't make the next one look late. A stop is unexpected if it's more than 20 % of the cycle
off the grid, or lasts much longer or shorter than usual. It needs at least 4 stops to learn; with fewer, it says so.

**Q: How is the machine stop detected?**
An "activity zone" over the machine: motion energy inside it each sample, auto-calibrated running vs stopped levels,
and blips shorter than 1.5 s dropped.

**Q: What if the LLM is wrong?**
Three safeguards: its code only reads the event log; cited event IDs are validated; and the rule engine answers the
same question, so if they disagree we show the evidence-backed answer. In testing it caught a weak model saying
"0 stops" and replaced it with 2. On our real clips, all 9 LLM answers agreed with the rule engine.

**Q: What happens with a long video?**
The sampling rate drops automatically. A 10-minute video takes about 2 min 15 s on our RTX 5070 laptop. Analyses are
cached, so questions afterwards are instant.

**Q: What didn't you finish / what are the limits?**
Single camera only (no cross-camera re-ID); identity for very small, distant people relies on the vision-model check;
actions beyond movement, zones, objects and sounds need the optional VLM captions; a camera that walks far away
from the reference view loses zone registration.

**Q: How would you make money? Who is the customer?**
Factories, warehouses, construction, retail, colleges and security companies, who today pay people to search CCTV.
We'd sell a subscription per camera, add-ons for real-time alerts and incident reports, and enterprise plans with
many cameras, an API and on-premise deployment, so footage never leaves the site. The report feature already works.

**Q: Why would they trust an AI for incidents?**
Because every answer is backed by timestamped events and evidence frames, it's cross-checked by a rule engine, and
it says "not observed" instead of guessing. A report that can't be verified is useless; ours can be.

**Q: Which parts did you build vs. use?**
Used: YOLO11 + BoT-SORT (ultralytics), ResNet-18 weights, OpenCV ORB/RANSAC, Gemini/Claude API, Streamlit. Built:
re-identification and VLM verification, camera registration, left-object detection, the event engine (12 event
types + audio), the temporal QA layer with validation and cross-check, the offline rule engine, the benchmark
generator and scorer, and the UI.

## If something breaks live

| Problem | Do this |
|---|---|
| Wi-Fi / API down or "quota" error | Switch the answer mode to **Offline**. Same answers for the demo questions. |
| A judge's video is slow to analyse | Settings → Detector `yolo11n.pt`, samples per second 5 |
| Wrong zone | **Zones** tab → drag a new box on the frame → Add zone → Analyse again |
| A judge brings their own video | Upload video → draw their "restricted area" (and an *activity* box over any machine) → Analyse |
| App crashes | Re-run the start command; analyses and answers are cached on disk |
