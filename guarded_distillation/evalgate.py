"""Fast regression eval for CI: replay seed 0 with the recommended preset from the cached Haiku labels
(no API calls) and compare with the committed baseline.

    gdistill eval                    # run, print the table, exit 1 if a threshold is broken
    gdistill eval --write-baseline   # record the current numbers in docs/eval_baseline.json

Thresholds: overall accuracy vs LLM only >= -0.3 pts, launch-question accuracy vs LLM only >= -2 pts,
LLM calls at most 2 pts above the baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .compat import load_pickle

BASELINE = Path("docs/eval_baseline.json")
LABELS = "labels/claude-haiku-5-5.jsonl"
MIN_OVERALL, MIN_LAUNCH, MAX_CALLS_OVER = -0.3, -2.0, 2.0


def metrics(run: Path) -> dict:
    res = {}
    for m in ("llm_only", "retrain"):
        with open(run / "results" / f"{m}.pkl", "rb") as f:
            c = load_pickle(f).calls
        res[m] = c[c.phase != "history"]
    a, b = res["retrain"], res["llm_only"]
    return {
        "calls_pct": round(float(a.llm_call.mean() * 100), 2),
        "overall_delta": round(float((a.correct.mean() - b.correct.mean()) * 100), 2),
        "launch_delta": round(float((a[a.is_launch].correct.mean() - b[b.is_launch].correct.mean()) * 100), 2),
        "accuracy_pct": round(float(a.correct.mean() * 100), 2),
        "llm_only_accuracy_pct": round(float(b.correct.mean() * 100), 2),
        "errors_per_1k": round(float(((a.decision == "accept") & ~a.correct).mean() * 1000), 1),
    }


def check(cur: dict, base: dict | None) -> list[tuple[str, str, str, bool]]:
    rows = [("Overall accuracy vs LLM only", f"{cur['overall_delta']:+.2f} pts", f">= {MIN_OVERALL:+.1f}", cur["overall_delta"] >= MIN_OVERALL),
            ("Launch-question accuracy vs LLM only", f"{cur['launch_delta']:+.2f} pts", f">= {MIN_LAUNCH:+.1f}", cur["launch_delta"] >= MIN_LAUNCH)]
    if base:
        lim = base["calls_pct"] + MAX_CALLS_OVER
        rows.append(("LLM calls", f"{cur['calls_pct']:.1f}%", f"<= {lim:.1f}% (baseline {base['calls_pct']:.1f}% + {MAX_CALLS_OVER:g})", cur["calls_pct"] <= lim))
    return rows


def markdown(cur: dict, base: dict | None, rows) -> str:
    out = ["## Eval replay (seed 0, recommended preset, cached Haiku labels)", "",
           "| Check | This run | Limit | Result |", "|---|---|---|---|"]
    out += [f"| {n} | {v} | {lim} | {'pass' if ok else '**FAIL**'} |" for n, v, lim, ok in rows]
    out += ["", f"Accuracy {cur['accuracy_pct']:.1f}% against {cur['llm_only_accuracy_pct']:.1f}% for the LLM alone; "
            f"{cur['errors_per_1k']:.0f} cheap-path errors per 1,000 queries."]
    return "\n".join(out)


def cmd_eval(args) -> None:
    from . import cli
    run = Path(args.run)
    ns = argparse.Namespace(run=str(run), seed=0, history=4000, n=8000, shots=1, acc=0.93, price="haiku-5.5", raw=args.raw,
                            teacher="anthropic", model="claude-haiku-5-5", base_url=None, workers=1, yes=False, labels=LABELS)
    cli.cmd_label(ns)
    cli.cmd_replay(argparse.Namespace(run=str(run), price=None, methods="llm_only,retrain", map="tfidf", preset="recommended",
                                      student=None, config=None, raw=args.raw))
    cur = metrics(run)
    if args.write_baseline:
        BASELINE.write_text(json.dumps({"seed": 0, "preset": "recommended", **cur}, indent=2) + "\n")
        print(f"wrote {BASELINE}: {cur}")
        return
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    rows = check(cur, base)
    md = markdown(cur, base, rows)
    print("\n" + md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")
    if not all(ok for *_, ok in rows):
        sys.exit("eval failed: " + "; ".join(n for n, _, _, ok in rows if not ok))
