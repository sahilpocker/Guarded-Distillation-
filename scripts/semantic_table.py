"""Markdown tables for the semantic-cortisol experiment, from runs/semantic/*.json (scripts/semantic.py)."""
import json
import sys
from pathlib import Path

import numpy as np

COLS = [("calls", "LLM calls, %", 1), ("acc_delta", "Overall Δ, pts", 2), ("launch_delta", "Launch Δ, pts", 2),
        ("launch600_delta", "Launch Δ, first 600 launch-phase queries, pts", 2), ("errors_per_1k", "Cheap-path errors / 1k", 1),
        ("extra_calls_pct", "Extra LLM calls on probation, % of queries", 1)]


def table(rows, seeds):
    print("| Setting | " + " | ".join(c[1] for c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for name, tag, m in rows:
        o = [json.loads(Path(f"runs/semantic/{tag}-s{k}.json").read_text())[m] for k in seeds]
        cells = []
        for key, _, d in COLS:
            v = [x[key] for x in o]
            cells.append(f"{np.mean(v):.{d}f} [{min(v):.{d}f} to {max(v):.{d}f}]")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    print("Seeds 0-2 (selection)")
    table([("Plain gate, no balancing (α 0.03)", "nobal", "retrain"), ("Plain gate + balanced (α 0.03)", "bal03", "retrain"),
           ("Plain gate + balanced (α 0.0175, calls-matched)", "bal0175", "retrain"),
           ("Plain gate + balanced (α 0.0125)", "bal0125", "retrain"),
           ("Semantic cortisol", "g07h4", "semantic")], [0, 1, 2])
    print("\nSeeds 3-5 (held out)")
    table([("Plain gate, no balancing (α 0.03)", "H-nobal", "retrain"), ("Plain gate + balanced (α 0.03)", "H-bal03", "retrain"),
           ("Plain gate + balanced (α 0.02)", "H-bal02", "retrain"),
           ("Plain gate + balanced (α 0.0175, calls-matched)", "H-bal0175", "retrain"),
           ("Plain gate + balanced (α 0.0125)", "H-bal0125", "retrain"),
           ("Semantic cortisol", "H-g07h4", "semantic"), ("Semantic cortisol + LLM sensor (relaxed trigger)", "S2", "semantic_llm")], [3, 4, 5])
