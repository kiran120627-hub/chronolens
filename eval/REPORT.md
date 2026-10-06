# Benchmark — synthetic warehouse video with exact ground truth

**Score: 13.0 / 13** (judging rule: right answer + right time = 1, right answer with wrong time = 0.5; time tolerance ±1.5 s) · hybrid: gemini-3.8-flash via generativelanguage.googleapis.com + rule engine

Video 1:40.0 analysed in 26s · people found 3 / 3 expected · 6 re-identification links · 36 events

| id | skill | question | score | answer |
|---|---|---|---|---|
| S01 | order / after | Which person entered the restricted area after the delivery truck arrived, and when? | ✅ 1.0 | After the delivery truck arrived at 26.48s (event E007), P2 entered the Restricted area at 40.64s (event E015), and P2 entered the Restricte |
| S02 | counting over time | How many times did the machine stop? | ✅ 1.0 | It stopped 2 time(s): 0:20.1–0:27.0, 1:00.1–1:04.0. |
| S03 | cause & effect / before | What happened right before the safety alarm went off? | ✅ 1.0 | Right before the safety alarm went off at 1:10.0, P2 entered the Restricted area at 1:06.0 and was inside the zone during the alarm. |
| S04 | event time | When did the alarm go off? | ✅ 1.0 | The alarm went off starting at 1:10.0 (events: E028, E029). |
| S05 | duration | When did the delivery vehicle arrive and how long did it stay parked? | ✅ 1.0 | The delivery vehicle (V1, bus) arrived at 0:26.5 (event E007) and stayed parked/stationary for 51.6 seconds, from 0:29.8 to 1:21.4 (event E0 |
| S06 | re-identification | Did the first person who appeared come back later? If so, when? | ✅ 1.0 | Yes, the first person who appeared (P1) came back later at 0:41.0. |
| S07 | stationary | Who stood still for more than 20 seconds, and when? | ✅ 1.0 | The following subjects stood still for more than 20 seconds: V1 stood still from 0:29.8 to 1:21.4 (51.6s); P3 stood still from 0:57.6 to 1:3 |
| S08 | identity tracking | How many different people appear in the video? | ✅ 1.0 | There are 3 different people that appear in the video: P1, P2, P3, appearing at 3.44, 33.44, 55.52s respectively. |
| S09 | ordering | In what order did people first enter the restricted area? | ✅ 1.0 | People entered the Restricted area in the following order: P1 at 0:12.0, P2 at 0:40.6, P2 at 1:06.0 |
| S10 | temporal overlap | Did anyone enter the restricted area while the machine was stopped? | ✅ 1.0 | No, nobody entered the restricted area while the machine was stopped. |
| S11 | duration aggregation | How long in total did the second person spend inside the restricted area? | ✅ 1.0 | P2 spent a total of 14.6 seconds inside the Restricted area across two intervals: from 0:40.6 to 0:46.4 and from 1:06.0 to 1:14.9. |
| S13 | audio event | Was there any loud crash or bang? When? | ✅ 1.0 | Yes, there was a loud sound (bang / crash / shout) at 0:35.0. |
| S12 | refuse unsupported | When did a dog run across the floor? | ✅ 1.0 | No dog was observed running across the floor in the video. |
