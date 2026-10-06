"""Score ChronoLens on the ground-truth benchmark video, using the problem statement's rule:

    correct answer + correct time (within ±tolerance)  -> 1.0
    correct answer, wrong / missing time               -> 0.5
    wrong answer                                       -> 0

    python data/make_synthetic.py      # once: renders the video + questions with exact ground truth
    python eval/run_eval.py            # analyse + answer + score  -> eval/results.json, eval/REPORT.md
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chronolens.config import Settings, Zone, fmt_t  # noqa: E402
from chronolens.llm import LLM  # noqa: E402
from chronolens.pipeline import analyze  # noqa: E402
from chronolens.qa import ask  # noqa: E402

YES = re.compile(r"\b(yes|did)\b", re.I)
NO = re.compile(r"\b(no|nobody|no one|none|did not|didn't)\b", re.I)


def grade(q: dict, r, tol: float) -> tuple[float, str, bool, bool]:
    exp = q["expected"]
    if exp.get("not_observed"):
        ok = r.status == "not_observed"
        return (1.0 if ok else 0.0), ("correctly reported not observed" if ok else "claimed to observe it"), ok, True
    if r.status != "answered":
        return 0.0, f"no answer ({r.status})", False, False
    ok, why = True, []
    if "subjects" in exp:
        kinds = {e[0] for e in exp["subjects"]}  # compare like with like (people vs vehicles)
        subj = [s for s in r.subjects if s[:1] in kinds]
        if len(exp["subjects"]) > 1:  # order matters
            got = [s for s in subj if s in exp["subjects"]]
            ok &= got[:len(exp["subjects"])] == exp["subjects"]
        else:
            ok &= exp["subjects"][0] in subj and len(set(subj) - {exp["subjects"][0]}) == 0
        why.append(f"subjects {subj} vs {exp['subjects']}")
    if "value" in exp:
        v = exp["value"]
        if isinstance(v, str):
            text = f"{r.value} {r.answer}"
            hit = bool(NO.search(text)) if v == "no" else bool(YES.search(text))
            ok &= hit
        else:
            try:
                got = float(r.value)
            except (TypeError, ValueError):
                nums = re.findall(r"-?\d+(?:\.\d+)?", r.answer.replace(",", ""))
                got = float(nums[0]) if nums else float("nan")
            ok &= abs(got - v) <= exp.get("value_tol", 0.01 if float(v).is_integer() else 0.5)
        why.append(f"value {r.value!r} vs {v!r}")
    times_ok = True
    if "times" in exp:
        for t in exp["times"]:
            times_ok &= any(abs(t - x) <= tol for x in r.timestamps)
        why.append("times " + ", ".join(fmt_t(x) for x in r.timestamps[:4]) + " vs " + ", ".join(fmt_t(t) for t in exp["times"]))
    score = (1.0 if times_ok else 0.5) if ok else 0.0
    return score, "; ".join(why), ok, times_ok


def main() -> None:
    meta = json.loads((ROOT / "eval" / "questions_sim.json").read_text())
    settings = Settings(zones=[Zone(z["name"], [tuple(p) for p in z["points"]], z["kind"]) for z in meta["zones"]])
    llm = LLM.from_env()
    if not llm.configured:
        sys.exit("No LLM key configured (.env)")
    t0 = time.time()
    a = analyze(ROOT / meta["video"], settings, progress=lambda f, m: print(f"  [{f:4.0%}] {m}", flush=True))
    t_an = time.time() - t0
    people = a.identities[a.identities.kind == "person"]
    print(f"\nidentities: {list(a.identities.id)}  (expected people: 3)")
    rows = []
    for q in meta["questions"]:
        t1 = time.time()
        try:
            r = ask(a, q["question"], llm)
        except Exception as e:  # noqa: BLE001 - one failed call must not stop the benchmark
            from chronolens.qa import QAResult
            r = QAResult(question=q["question"], status="error", error=str(e)[:300])
        s, why, ans_ok, time_ok = grade(q, r, meta["time_tolerance"])
        rows.append({"id": q["id"], "skill": q["skill"], "question": q["question"], "score": s, "answer": r.answer,
                     "status": r.status, "timestamps": r.timestamps, "subjects": r.subjects, "value": r.value,
                     "confidence": r.confidence, "why": why, "seconds": round(time.time() - t1, 1)})
        print(f"{q['id']} {s:.1f}  {r.status:<12} {(r.answer or r.error)[:150]}")
        print(f"        grader: {why}")
    total = sum(r["score"] for r in rows)
    summary = {"score": total, "max": len(rows), "model": llm.label, "analysis_seconds": round(t_an, 1),
               "video_seconds": a.meta["duration"], "people_found": len(people), "people_expected": 3,
               "reid_links": len(a.links), "events": len(a.events)}
    (ROOT / "eval" / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str))
    lines = ["# Benchmark — synthetic warehouse video with exact ground truth", "",
             f"**Score: {total:.1f} / {len(rows)}** (judging rule: right answer + right time = 1, right answer with "
             f"wrong time = 0.5; time tolerance ±{meta['time_tolerance']} s) · model {llm.label}", "",
             f"Video {fmt_t(a.meta['duration'])} analysed in {t_an:.0f}s · people found {len(people)} / 3 expected · "
             f"{len(a.links)} re-identification links · {len(a.events)} events", "",
             "| id | skill | question | score | answer |", "|---|---|---|---|---|"]
    for r in rows:
        icon = "✅" if r["score"] == 1 else ("🟡" if r["score"] == 0.5 else "❌")
        lines.append(f"| {r['id']} | {r['skill']} | {r['question']} | {icon} {r['score']} | {r['answer'][:140]} |")
    (ROOT / "eval" / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSCORE {total:.1f}/{len(rows)}")


if __name__ == "__main__":
    main()
