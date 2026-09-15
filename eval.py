from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from db import run_query                      
from tools import compare_results as compare              
from graph import run_agent                   

QUESTIONS_PATH = Path("eval_questions.json")

def evaluate_one(
    q: dict,
    max_attempts: int,
    use_domain_notes: bool = True,
    schema_mode: str = "full",
    cross_check: bool = False,
    network_retries: int = 3,
) -> dict:
                                                                               
    gold_cols, gold_rows = run_query(q["gold_sql"])

    last_error: Exception | None = None
    state = None
    for attempt in range(network_retries):
        try:
            state = run_agent(
                q["question"],
                max_attempts=max_attempts,
                use_domain_notes=use_domain_notes,
                schema_mode=schema_mode,
                cross_check=cross_check,
            )
            break
        except Exception as e:
            last_error = e
            if attempt < network_retries - 1:
                wait = 2 ** attempt * 3
                print(f"        network error, retrying in {wait}s: {str(e)[:70]}")
                time.sleep(wait)

    if state is None:
        e = last_error
        return {
            **{k: q.get(k) for k in ("id", "category", "question", "gold_sql")},
            "crashed": True, "crash_error": str(e)[:300],
            "executed": False, "strict_match": False, "lenient_match": False,
            "attempts": 0, "first_attempt_failed": None,
            "sql": "", "answer": "", "history": [], "elapsed_s": 0.0,
            "agreement": "skipped", "confidence": "unverified",
        }

    executed = state["status"] == "success"
    strict = lenient = False
    if executed:
        strict, lenient = compare(gold_rows, state["rows"])

    history = state.get("history", [])
    first_failed = bool(history) and history[0]["stage"] != "success"

    return {
        **{k: q.get(k) for k in ("id", "category", "question", "gold_sql")},
        "crashed": False, "crash_error": None,
        "sql": state["sql"],
        "answer": state["answer"],
        "executed": executed,
        "strict_match": strict,
        "lenient_match": lenient,
        "attempts": state["attempts"],
        "first_attempt_failed": first_failed,
        "gold_row_count": len(gold_rows),
        "agent_row_count": len(state["rows"]),
        "history": history,
        "elapsed_s": state["elapsed_s"],
        "agreement": state.get("agreement", "skipped"),
        "confidence": state.get("confidence", "unverified"),
        "cross_check_sql": state.get("cross_check_sql", ""),
    }

def summarise(results: list[dict]) -> dict:

    crashed = [r for r in results if r["crashed"]]
    scored = [r for r in results if not r["crashed"]]

    n = len(scored)
    pct = lambda c: round(100 * c / n, 1) if n else 0.0

    executed = sum(r["executed"] for r in scored)
    strict = sum(r["strict_match"] for r in scored)
    lenient = sum(r["lenient_match"] for r in scored)

    needed_retry = [r for r in scored if r["first_attempt_failed"]]
    rescued = [r for r in needed_retry if r["executed"]]
    rescued_correct = [r for r in needed_retry if r["lenient_match"]]

    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "executed": 0, "correct": 0})
    for r in scored:
        c = by_cat[r["category"]]
        c["n"] += 1
        c["executed"] += r["executed"]
        c["correct"] += r["lenient_match"]

    verified = [r for r in scored if r.get("agreement") in ("agree", "disagree")]
    wrong = [r for r in verified if not r["lenient_match"]]
    right = [r for r in verified if r["lenient_match"]]
    caught = [r for r in wrong if r["agreement"] == "disagree"]
    false_alarms = [r for r in right if r["agreement"] == "disagree"]

    return {
        "questions_attempted": len(results),
        "questions_scored": n,
        "crashed": len(crashed),
        "crashed_ids": [r["id"] for r in crashed],
        "execution_accuracy_pct": pct(executed),
        "answer_accuracy_strict_pct": pct(strict),
        "answer_accuracy_lenient_pct": pct(lenient),
        "avg_attempts": round(sum(r["attempts"] for r in scored) / n, 2) if n else 0,
        "questions_needing_retry": len(needed_retry),
        "retry_execution_recovery_pct": (
            round(100 * len(rescued) / len(needed_retry), 1) if needed_retry else None
        ),
        "retry_answer_recovery_pct": (
            round(100 * len(rescued_correct) / len(needed_retry), 1) if needed_retry else None
        ),
        "avg_seconds_per_question": (
            round(sum(r["elapsed_s"] for r in scored) / n, 2) if n else 0
        ),
        "by_category": {k: dict(v) for k, v in sorted(by_cat.items())},

        "cross_check_verified": len(verified),
        "cross_check_inconclusive": sum(
            1 for r in scored if r.get("agreement") == "inconclusive"
        ),
        "wrong_answers": len(wrong),
        "wrong_answers_flagged": len(caught),
        "wrong_answers_flagged_pct": (
            round(100 * len(caught) / len(wrong), 1) if wrong else None
        ),
        "false_alarm_pct": (
            round(100 * len(false_alarms) / len(right), 1) if right else None
        ),
        "flagged_ids": [r["id"] for r in verified if r["agreement"] == "disagree"],
    }

def print_report(results: list[dict], summary: dict) -> None:
    w = (6, 14, 40, 5, 5, 8, 8)
    header = ("ID", "CATEGORY", "QUESTION", "RUN", "OK", "ATTEMPTS", "TIME")
    line = "-" * (sum(w) + len(w) * 2)

    print("\n" + line)
    print("PER-QUESTION RESULTS")
    print(line)
    print("  ".join(h.ljust(x) for h, x in zip(header, w)))
    print(line)

    for r in results:
        q = r["question"]
        cells = (
            r["id"],
            r["category"],
            (q[:37] + "...") if len(q) > 40 else q,
            "yes" if r["executed"] else "NO",
            "yes" if r["lenient_match"] else "NO",
            str(r["attempts"]),
            f"{r['elapsed_s']}s",
        )
        print("  ".join(str(c).ljust(x) for c, x in zip(cells, w)))

    print(line)
    print("SUMMARY")
    print(line)
    s = summary
    print(f"  Questions scored               {s['questions_scored']} of {s['questions_attempted']}")
    print(f"  Execution accuracy             {s['execution_accuracy_pct']}%")
    print(f"  Answer accuracy (lenient)      {s['answer_accuracy_lenient_pct']}%   <- headline number")
    print(f"  Answer accuracy (strict)       {s['answer_accuracy_strict_pct']}%")
    print(f"  Average attempts per question  {s['avg_attempts']}")
    print(f"  Questions needing a retry      {s['questions_needing_retry']}")
    print(f"  Retry recovery (executed)      {s['retry_execution_recovery_pct']}%")
    print(f"  Retry recovery (correct)       {s['retry_answer_recovery_pct']}%")
    print(f"  Average time per question      {s['avg_seconds_per_question']}s")

    if s["cross_check_verified"]:
        print(f"\n  Cross-check (catches wrong-but-runnable answers)")
        print(f"    Questions verified           {s['cross_check_verified']}"
              f"  ({s['cross_check_inconclusive']} inconclusive)")
        print(f"    Wrong answers                {s['wrong_answers']}")
        print(f"    ...flagged as disagreement   {s['wrong_answers_flagged']}"
              f"  ({s['wrong_answers_flagged_pct']}%)   <- detection rate")
        print(f"    False alarms on correct ones {s['false_alarm_pct']}%")
        if s["flagged_ids"]:
            print(f"    Flagged: {', '.join(s['flagged_ids'])}")
    if s["crashed"]:
        print(f"\n  !! {s['crashed']} question(s) never reached the model and are EXCLUDED")
        print(f"     from every percentage above: {', '.join(s['crashed_ids'])}")
        print("     These are network or API failures, not agent errors. Re-run to")
        print("     score them. If they persist, check your connection or API key.")

    print("\n  By category:")
    for cat, v in s["by_category"].items():
        print(f"    {cat:14} {v['correct']}/{v['n']} correct, {v['executed']}/{v['n']} executed")

    failures = [r for r in results if not r["lenient_match"] and not r["crashed"]]
    if failures:
        print(f"\n  Failed questions ({len(failures)}):")
        for r in failures:
            reason = r["crash_error"] or (
                "wrong result" if r["executed"] else
                (r["history"][-1]["error"] if r["history"] else "unknown")
            )
            print(f"    [{r['id']}] {r['question']}")
            print(f"          -> {str(reason)[:120]}")
    print(line + "\n")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between questions (Groq free tier is rate limited)")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--questions", default="eval_questions.json",
                   help="which test set to run, e.g. eval_questions_hard.json")
    p.add_argument("--db", default=None,
                   help="database file to query, e.g. chinook_dirty.db. The gold "
                        "SQL in --questions must match this schema.")
    p.add_argument("--model", default=None,
                   help="override GROQ_MODEL for this run, e.g. llama-3.1-8b-instant. "
                        "A smaller model fails more often, which is what makes the "
                        "retry loop measurable.")
    p.add_argument("--schema-mode", default="full",
                   choices=["full", "no_fk", "names_only"],
                   help="how much schema the model sees. 'no_fk' hides foreign keys "
                        "so it must guess join columns; 'names_only' hides columns "
                        "entirely. Both manufacture real, recoverable failures.")
    p.add_argument("--cross-check", action="store_true",
                   help="run a second independent query per question and compare "
                        "the results. Costs one extra LLM call each; catches "
                        "queries that run cleanly but return the wrong answer, "
                        "which the retry loop cannot.")
    p.add_argument("--handicap", action="store_true",
                   help="strip the hand-written DOMAIN_NOTES from the generator "
                        "prompt. Use this to measure self-correction: with the "
                        "hints in place the model rarely fails, so the retry "
                        "loop never runs and retry metrics stay empty.")
    args = p.parse_args()

    if args.model:
        os.environ["GROQ_MODEL"] = args.model
    if args.db:
        os.environ["CHINOOK_DB_PATH"] = args.db

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]

    print(f"\nTest set:  {args.questions}"
          f"\nModel:     {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}"
          f"\nDatabase:  {os.getenv('CHINOOK_DB_PATH', 'chinook.db')}"
          f"\nSchema:    {args.schema_mode}"
          f"\nHints:     {'OFF (handicap)' if args.handicap else 'on'}"
          f"\nCross-chk: {'on' if args.cross_check else 'off'}"
          f"\nAttempts:  {args.max_attempts}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question']}")
        r = evaluate_one(q, args.max_attempts,
                         use_domain_notes=not args.handicap,
                         schema_mode=args.schema_mode,
                         cross_check=args.cross_check)
        flag = "OK " if r["lenient_match"] else "BAD"
        extra = ""
        if r["agreement"] not in ("skipped",):
            extra = f" cross-check={r['agreement']}"
        print(f"        {flag} attempts={r['attempts']} rows={r.get('agent_row_count')} "
              f"(gold {r.get('gold_row_count')}){extra}")
        results.append(r)
        if args.sleep:
            time.sleep(args.sleep)

    summary = summarise(results)
    summary["handicap"] = args.handicap
    summary["test_set"] = args.questions
    summary["max_attempts"] = args.max_attempts
    summary["model"] = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    summary["schema_mode"] = args.schema_mode
    summary["database"] = os.getenv("CHINOOK_DB_PATH", "chinook.db")
    summary["cross_check"] = args.cross_check
    print_report(results, summary)

    if summary["questions_needing_retry"] == 0:
        print("  NOTE: no question failed on attempt 1, so the retry loop never ran")
        print("        and the recovery metrics above are empty. This measures the")
        print("        generator, not self-correction. A strong model with the full")
        print("        schema rarely fails on Chinook, so degrade the conditions:")
        print("          python eval.py --model llama-3.1-8b-instant --schema-mode names_only")
        print("          python eval.py --schema-mode names_only --handicap\n")

    Path(args.out).write_text(
        json.dumps(
            {"summary": summary, "results": results, "run_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )

    csv_lines = ["id,category,executed,answer_correct_lenient,answer_correct_strict,"
                 "attempts,seconds,agreement"]
    csv_lines += [
        f"{r['id']},{r['category']},{int(r['executed'])},{int(r['lenient_match'])},"
        f"{int(r['strict_match'])},{r['attempts']},{r['elapsed_s']},{r.get('agreement','skipped')}"
        for r in results
    ]
    csv_path = Path(args.out).with_suffix(".csv")
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")

    print(f"Saved {args.out} and {csv_path}")

if __name__ == "__main__":
    main()
