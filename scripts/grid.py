"""Run replays for the operating-point grid, reusing runs that already exist.

    python scripts/grid.py                       # seeds 0-2, all students and alphas, retrain/global/cortisol
    python scripts/grid.py --seeds 3 4 5 --students minilm --alphas 0.05 --methods llm_only,global

A cell is (student, alpha_intent, seed); the map space is always TF-IDF so the student effect is
isolated. Teacher labels are read from the shared Haiku cache (no API calls).
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LABELS = "labels/claude-haiku-5-5.jsonl"
STUDENT_FLAG = {"tfidf": "tfidf", "minilm": "sbert"}
# Runs made before the grid existed (TF-IDF student, TF-IDF map, same engine settings).
EXISTING = {("tfidf", 0.03, k): f"runs/haiku-s{k}" for k in (0, 1, 2)}
EXISTING |= {("tfidf", 0.05, k): f"runs/haiku-a05-s{k}" for k in (0, 1, 2)}
EXISTING |= {("tfidf", 0.08, 0): "runs/haiku-a08-s0"}


def run_dir(student: str, alpha: float, seed: int) -> str:
    return EXISTING.get((student, alpha, seed)) or f"runs/grid-{student}-a{round(alpha * 100):02d}-s{seed}"


def done(d: str, methods: list[str]) -> bool:
    return (Path(d) / "results.json").exists() and all((Path(d) / "results" / f"{m}.pkl").exists() for m in methods)


def run(cell, methods: list[str]) -> str:
    student, alpha, seed = cell
    d = run_dir(*cell)
    if done(d, methods):
        return f"reuse {d}"
    log = open(d + ".log", "w") if Path(d).parent.exists() else sys.stdout
    sh = lambda *a: subprocess.run(["cortisol", *a], stdout=log, stderr=subprocess.STDOUT, check=True)  # noqa: E731
    if not (Path(d) / "teacher.csv").exists():
        sh("label", "--teacher", "anthropic", "--model", "claude-haiku-5-5", "--labels", LABELS,
           "--run", d, "--seed", str(seed), "--price", "haiku-5.5")
    sh("replay", "--run", d, "--map", "tfidf", "--student", STUDENT_FLAG[student], "--methods", ",".join(methods),
       "--config", json.dumps({"alpha_intent": alpha}))
    sh("report", "--run", d)
    return f"done  {d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--students", nargs="+", default=["tfidf", "minilm"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.05, 0.08])
    ap.add_argument("--methods", default="llm_only,retrain,global,cortisol")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    methods = a.methods.split(",")
    cells = [(s, al, k) for k in a.seeds for s in a.students for al in a.alphas]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda c: run(c, methods), cells):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
