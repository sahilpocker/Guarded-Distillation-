"""Markdown results table over seeds, from the REPORT.md of each run.

    python scripts/seed_table.py runs/haiku-a05-s0 runs/haiku-a05-s1 runs/haiku-a05-s2
"""
import re
import sys
from pathlib import Path

import numpy as np

NAMES = ["LLM only", "Static gate", "Static gate + retraining", "Global cortisol", "Cortisol field"]
num = lambda s: float(re.sub(r"[^0-9.\-+]", "", s))   # noqa: E731


def main(runs: list[str]) -> None:
    rows = {n: [] for n in NAMES}
    for r in runs:
        for line in (Path(r) / "REPORT.md").read_text().splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if c[0] in rows and len(c) == 9 and "%" in c[1]:
                rows[c[0]].append(c)
    base = np.array([num(r[1]) for r in rows["LLM only"]])
    print("| Method | Accuracy (" + " / ".join(f"s{i}" for i in range(len(runs))) + ") | Δ vs LLM only, mean | LLM calls, mean | "
          "Launch questions, mean | Off-topic questions, mean | Cost per 1,000 queries, mean | Cheap-path errors / 1k, mean |")
    print("|---|---|---|---|---|---|---|---|")
    for n in NAMES:
        R = rows[n]
        acc = np.array([num(r[1]) for r in R])
        d = "baseline" if n == "LLM only" else f"{np.mean(acc - base):+.2f} pts"
        print(f"| {n} | {' / '.join('%.1f' % a for a in acc)} | {d} | {np.mean([num(r[3]) for r in R]):.0f}% | "
              f"{np.mean([num(r[6]) for r in R]):.1f}% | {np.mean([num(r[7]) for r in R]):.1f}% | "
              f"${np.mean([num(r[4]) for r in R]):.4f} | {np.mean([num(r[5]) for r in R]):.1f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
