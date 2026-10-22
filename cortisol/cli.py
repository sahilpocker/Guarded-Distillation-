"""Command line: cortisol data | label | replay | report | demo

    cortisol demo                                   # everything, simulated teacher, free
    cortisol label --teacher anthropic --model claude-haiku-5-5 --run runs/haiku-s0 --yes
    cortisol replay --run runs/haiku-s0
    cortisol report --run runs/haiku-s0
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .engine import METHODS, EngineConfig, config_dict, run_method
from .report import write_all
from .teacher import PRICES, CallCost, LLMTeacher, SimulatedTeacher, approx_tokens, build_system_prompt, precompute

ROOT = Path.cwd()


def _run_dir(args) -> Path:
    return Path(args.run)


def cmd_data(args) -> None:
    raw = D.download(args.raw)
    pools = D.load_pools(raw)
    print(f"ok: {len(pools.banking):,} Banking77 queries, {len(pools.oos):,} out-of-scope queries in {raw}")


def cmd_label(args) -> None:
    run = _run_dir(args)
    run.mkdir(parents=True, exist_ok=True)
    pools = D.load_pools(args.raw)
    sc = D.StreamConfig(history=args.history, n=args.n, seed=args.seed, shots=args.shots)
    stream, fewshot = D.build_stream(pools, sc)
    system_prompt = build_system_prompt(pools.intents, fewshot)
    if args.teacher == "simulated":
        teacher = SimulatedTeacher(pd.concat([pools.banking, pools.oos]), acc=args.acc, seed=args.seed)
        name = f"simulated (acc {args.acc:.0%})"
    else:
        if not args.model:
            sys.exit("--model is required for a real teacher")
        n_calls = len(stream)
        sys_tok = approx_tokens(system_prompt)
        price = PRICES.get(args.price)
        est = n_calls * (sys_tok + 25) * price.input / 1e6 + n_calls * 8 * price.output / 1e6 if price else None
        print(f"{n_calls:,} teacher calls; system prompt ~{sys_tok:,} tokens.")
        if est is not None:
            print(f"Upper-bound cost at {price.name} prices without prompt caching: ${est:.2f}")
        if not args.yes:
            sys.exit("Add --yes to make the calls. Answers are cached in the run folder, so a rerun costs nothing.")
        teacher = LLMTeacher(args.teacher, args.model, system_prompt, list(pools.intents) + [D.OOS],
                             cache_path=run / "teacher_cache.jsonl", base_url=args.base_url,
                             max_workers=args.workers)
        name = f"{args.teacher}:{args.model}"
    t0 = time.time()
    tdf = precompute(stream, teacher, system_prompt)
    stream.to_csv(run / "stream.csv", index=False)
    fewshot.to_csv(run / "fewshot.csv", index=False)
    tdf.to_csv(run / "teacher.csv", index=False)
    (run / "system_prompt.txt").write_text(system_prompt)
    live = (stream.phase != "history").to_numpy()
    acc = float((tdf.teacher.to_numpy() == stream.gold.to_numpy())[live].mean())
    meta = {"run": run.name, "teacher": name, "seed": args.seed, "history": args.history, "live": args.n,
            "teacher_accuracy_live": acc, "price": args.price}
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"labelled {len(stream):,} queries in {time.time() - t0:.0f}s; teacher accuracy vs human labels "
          f"(live part) {acc:.1%}; saved to {run}")


def _load_run(run: Path):
    stream = pd.read_csv(run / "stream.csv", keep_default_na=False)
    tdf = pd.read_csv(run / "teacher.csv", keep_default_na=False)
    meta = json.loads((run / "meta.json").read_text())
    stream["is_launch"] = stream.is_launch.astype(str).str.lower().eq("true")
    return stream, tdf, meta


def cmd_replay(args) -> None:
    run = _run_dir(args)
    stream, tdf, meta = _load_run(run)
    pools = D.load_pools(args.raw)
    profile = args.price or meta.get("price", "haiku-5.5")
    price = PRICES[profile]
    cost = np.array([CallCost(int(r.input_tokens), int(r.output_tokens), int(r.cache_read_tokens),
                              int(r.cache_write_tokens)).usd(price) for r in tdf.itertuples()])
    overrides = json.loads(args.config) if args.config else {}
    cfg = EngineConfig(seed=meta["seed"], map_backend=args.map, **overrides)
    out = run / "results"
    out.mkdir(exist_ok=True)
    for name in args.methods.split(","):
        t0 = time.time()
        try:
            res = run_method(stream, tdf, pools.intents, METHODS[name], cfg, cost)
        except NotImplementedError:
            sys.exit(f"\n{name}: cortisol.gate.aci_update is not written yet. It is yours (one line):\n"
                     "  read its docstring in cortisol/gate.py and docs/THEORY.md lesson 2, then\n"
                     "  python -m pytest -q tests/test_aci_spec.py   until it passes.")
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(res, f)
        c = res.calls[res.calls.phase != "history"]
        print(f"{name:>9}: accuracy {c.correct.mean():.1%}  LLM calls {c.llm_call.mean():.0%}  "
              f"({time.time() - t0:.0f}s)")
    meta["engine"] = config_dict(cfg)
    meta["price"] = profile
    meta["price_model"] = price.name
    (run / "meta.json").write_text(json.dumps(meta, indent=2))


def cmd_report(args) -> None:
    run = _run_dir(args)
    meta = json.loads((run / "meta.json").read_text())
    results = {}
    for name in ("llm_only", "static", "retrain", "global", "cortisol"):
        p = run / "results" / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                results[name] = pickle.load(f)
    if "cortisol" not in results:
        sys.exit("run `cortisol replay` first (the report needs at least the cortisol method)")
    write_all(results, meta, run)
    print((run / "REPORT.md").read_text())
    print(f"dashboard: {run / 'replay.html'}")


def cmd_demo(args) -> None:
    cmd_data(args)
    args.teacher = "simulated"
    cmd_label(args)
    args.methods = "llm_only,static,retrain,global,cortisol"
    args.config = None
    cmd_replay(args)
    cmd_report(args)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="cortisol", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw", help="where the datasets live")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("data", help="download Banking77 and CLINC150 (pinned hashes)")

    def stream_args(q):
        q.add_argument("--run", default="runs/simulated-s0", help="run folder")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--history", type=int, default=4000, help="shadow-mode queries before go-live")
        q.add_argument("--n", type=int, default=8000, help="live queries")
        q.add_argument("--shots", type=int, default=1, help="examples per label in the teacher prompt")
        q.add_argument("--acc", type=float, default=0.93, help="simulated teacher accuracy")
        q.add_argument("--price", default="haiku-5.5", choices=sorted(PRICES))

    q = sub.add_parser("label", help="label the stream with a teacher (cached)")
    stream_args(q)
    q.add_argument("--teacher", default="simulated", choices=["simulated", "anthropic", "openai"])
    q.add_argument("--model", default=None, help="e.g. claude-haiku-5-5, or a local model name")
    q.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--yes", action="store_true", help="really make the (paid) calls")

    def replay_args(q):
        q.add_argument("--methods", default="llm_only,static,retrain,global,cortisol")
        q.add_argument("--map", default="tfidf", choices=["tfidf", "sbert"], help="map space backend")
        q.add_argument("--config", default=None, help='JSON overrides for EngineConfig, e.g. \'{"alpha_intent":0.02}\'')

    q = sub.add_parser("replay", help="replay the stream for each method")
    q.add_argument("--run", default="runs/simulated-s0")
    q.add_argument("--price", default=None, choices=sorted(PRICES))
    replay_args(q)

    q = sub.add_parser("report", help="REPORT.md, results.json and replay.html for a run")
    q.add_argument("--run", default="runs/simulated-s0")

    q = sub.add_parser("demo", help="data + simulated labels + replay + report")
    stream_args(q)
    replay_args(q)

    args = p.parse_args(argv)
    {"data": cmd_data, "label": cmd_label, "replay": cmd_replay, "report": cmd_report, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    main()
