# Benchmark — synthetic warehouse video with exact ground truth

**Score: 13.0 / 13** (judging rule: right answer + right time = 1, right answer with wrong time = 0.5; time tolerance ±1.5 s) · offline rule engine (no LLM)

Video 1:40.0 analysed in 40s · people found 3 / 3 expected · 6 re-identification links · 36 events

| id | skill | question | score | answer |
|---|---|---|---|---|
| S01 | order / after | Which person entered the restricted area after the delivery truck arrived, and when? | ✅ 1.0 | Entered Restricted area after the vehicle V1 arrived at 0:26.5: P2 at 0:40.6, P2 at 1:06.0. |
| S02 | counting over time | How many times did the machine stop? | ✅ 1.0 | It stopped 2 time(s): 0:20.1–0:27.0, 1:00.1–1:04.0. |
| S03 | cause & effect / before | What happened right before the safety alarm went off? | ✅ 1.0 | Right before the alarm at 1:10.0: P2 enters Restricted area at 1:06.0 (4.0s earlier). |
| S04 | event time | When did the alarm go off? | ✅ 1.0 | The alarm went off at 1:10.0 (until 1:13.9): alarm-like tone / siren / beeper, +29 dB above background; red light / flashing. |
| S05 | duration | When did the delivery vehicle arrive and how long did it stay parked? | ✅ 1.0 | The vehicle V1 arrived at 0:26.5. It stayed parked 0:29.8–1:21.4 (51.7s). |
| S06 | re-identification | Did the first person who appeared come back later? If so, when? | ✅ 1.0 | Yes: P1 came back at 0:41.0 (seen again after 23.4s unseen (re-identified, link score 0.73)). |
| S07 | stationary | Who stood still for more than 20 seconds, and when? | ✅ 1.0 | Stood still >20s: P3 from 0:57.7 to 1:33.3 (35.6s). |
| S08 | identity tracking | How many different people appear in the video? | ✅ 1.0 | 3 different people appear: P1 (0:03.4–0:51.5), P2 (0:33.4–1:15.4), P3 (0:55.5–1:36.4). |
| S09 | ordering | In what order did people first enter the restricted area? | ✅ 1.0 | Order of first entry into Restricted area: P1 at 0:12.0, P2 at 0:40.6. |
| S10 | temporal overlap | Did anyone enter the restricted area while the machine was stopped? | ✅ 1.0 | No. Nobody entered Restricted area while the machine was stopped (stops: 0:20.1–0:27.0, 1:00.1–1:04.0). |
| S11 | duration aggregation | How long in total did the second person spend inside the restricted area? | ✅ 1.0 | P2 spent 14.6s in total inside Restricted area across 2 visit(s): 0:40.6–0:46.4, 1:06.0–1:14.9. |
| S13 | audio event | Was there any loud crash or bang? When? | ✅ 1.0 | Loud sound(s) at 0:35.0 (loud sound (bang / crash / shout), +31 dB above background). |
| S12 | refuse unsupported | When did a dog run across the floor? | ✅ 1.0 | Not observed: nothing matching 'dog' was detected in this video. |
