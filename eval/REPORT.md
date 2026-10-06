# Benchmark — synthetic warehouse video with exact ground truth

**Score: 12.0 / 12** (judging rule: right answer + right time = 1, right answer with wrong time = 0.5; time tolerance ±1.5 s) · model gemini-3.8-flash via generativelanguage.googleapis.com

Video 1:40.0 analysed in 0s · people found 3 / 3 expected · 6 re-identification links · 34 events

| id | skill | question | score | answer |
|---|---|---|---|---|
| S01 | order / after | Which person entered the restricted area after the delivery truck arrived, and when? | ✅ 1.0 | After the vehicle (V1) arrived at 0:26.5 (E007), person P2 entered the restricted area at 0:40.6 and again at 1:06.0 (E013, E024). |
| S02 | counting over time | How many times did the machine stop? | ✅ 1.0 | The machine stopped 2 times: from 0:20.1 to 0:27.0 (E006) and from 1:01.0 to 1:06.0 (E023). |
| S03 | cause & effect / before | What happened right before the safety alarm went off? | ✅ 1.0 | Right before the safety alarm (at 70.0s), the following events occurred: activity_stop at 61.0s (motion in Machine stops for 5.0s), activity |
| S04 | event time | When did the alarm go off? | ✅ 1.0 | The alarm went off at 1:10.0 (70.00s) and lasted until 1:13.8 (73.76s), indicated by flashing red light, sudden brightness changes, and a bu |
| S05 | duration | When did the delivery vehicle arrive and how long did it stay parked? | ✅ 1.0 | The delivery vehicle (V1) arrived at 0:26.5 (E007) and stayed parked for 51.6 seconds from 0:29.8 to 1:21.4 (E009). |
| S06 | re-identification | Did the first person who appeared come back later? If so, when? | ✅ 1.0 | Yes, the first person (P1), who originally entered at 0:03.4 (E001) and left at 0:17.5 (E005), came back later at 0:41.0 (E015). |
| S07 | stationary | Who stood still for more than 20 seconds, and when? | ✅ 1.0 | The following stood still for more than 20 seconds: V1 (bus) from 0:29.8 to 1:21.4 (for 51.6s), P3 (person) from 0:57.7 to 1:33.3 (for 35.6s |
| S08 | identity tracking | How many different people appear in the video? | ✅ 1.0 | There are 3 different people who appear in the video: P1 (seen 0:03.4 to 0:51.5), P2 (seen 0:33.4 to 1:15.4), P3 (seen 0:55.5 to 1:36.4). |
| S09 | ordering | In what order did people first enter the restricted area? | ✅ 1.0 | The people entered the restricted area in the following order: P1 at 0:12.0, P2 at 0:40.6 |
| S10 | temporal overlap | Did anyone enter the restricted area while the machine was stopped? | ✅ 1.0 | No one entered the restricted area strictly while the machine was stopped. The machine was stopped twice: from 0:20.1 to 0:27.0 (E006) and f |
| S11 | duration aggregation | How long in total did the second person spend inside the restricted area? | ✅ 1.0 | The second person (P2) spent a total of 14.6 seconds inside the Restricted area across two intervals: starting at 0:40.6 (event E014) and 1: |
| S12 | refuse unsupported | When did a dog run across the floor? | ✅ 1.0 | No dog or animal was detected or recorded in the video data. |
