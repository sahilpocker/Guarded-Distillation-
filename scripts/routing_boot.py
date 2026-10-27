"""Router vs margin on seeds 3-8 with paired bootstrap over queries (see scripts/boot.py).

    python scripts/routing_boot.py run      # margin and learned at the alphas chosen on seeds 0-2 for 40/50/60% calls, saving per-query arrays
    python scripts/routing_boot.py report   # tables for docs/RESULTS.md
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import boot  # noqa: E402
import routing as R  # noqa: E402

SEEDS = range(3, 9)
PAIR = ("learned", "margin")


def alpha(fam, router, b):
    sel = json.loads((R.OUT / "select.json").read_text())
    return round(sel[f"{fam}|{router}|{b}"], 5)


def run(workers=6):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(f, r, alpha(f, r, b), s) for f in R.FAMILIES for r in PAIR for b in R.BUDGETS for s in SEEDS]
    jobs = [j for j in jobs if not (R.path(*j).exists() and R.npz_path(*j).exists())]
    print(f"{len(jobs)} runs to do", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(R.run_one, *j, True) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            f.result()
            if n % 12 == 0:
                print(f"{n}/{len(jobs)}", flush=True)


def report():
    lines = ["| Stream | Budget | Overall accuracy: learned / margin | Difference, pts [95% CI] | New-product accuracy, first 600: learned / margin | Difference, pts [95% CI] | LLM calls: learned / margin |",
             "|---|---|---|---|---|---|---|"]
    for f in R.FAMILIES:
        for b in R.BUDGETS:
            arrs = {r: [np.load(R.npz_path(f, r, alpha(f, r, b), s)) for s in SEEDS] for r in PAIR}
            jl = {r: [json.loads(R.path(f, r, alpha(f, r, b), s).read_text()) for s in SEEDS] for r in PAIR}
            A, B = arrs[PAIR[0]], arrs[PAIR[1]]
            allq = boot.paired(A, B, "correct")
            lau = boot.paired([{**dict(a), "m": a["first"] & a["is_launch"]} for a in A], [{**dict(c), "m": c["first"] & c["is_launch"]} for c in B], "correct", ["m"])
            mean = lambda arrs_, key, mask=None: float(np.mean(np.concatenate([a[key][(a["first"] & a["is_launch"]) if mask else slice(None)] for a in arrs_])) * 100)  # noqa: E731
            lines.append(f"| {R.FAM_TITLE[f]} | {b}% | {mean(A, 'correct'):.2f}% / {mean(B, 'correct'):.2f}% | {boot.fmt(allq)} | {mean(A, 'correct', 1):.1f}% / {mean(B, 'correct', 1):.1f}% | "
                         f"{boot.fmt(lau)} | {np.mean([j['calls'] for j in jl[PAIR[0]]]):.1f}% / {np.mean([j['calls'] for j in jl[PAIR[1]]]):.1f}% |")
    out = "\n".join(lines)
    Path("runs/routing/boot_report.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
