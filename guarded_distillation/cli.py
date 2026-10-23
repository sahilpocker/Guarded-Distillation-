"""Command line: gdistill data | label | label-pool | replay | report | demo

    gdistill demo                                   # everything, simulated teacher, free
    gdistill label --teacher anthropic --model claude-haiku-5-5 --run runs/haiku-s0 --yes
    gdistill replay --run runs/haiku-s0
    gdistill report --run runs/haiku-s0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .compat import load_pickle
from .engine import METHODS, RECOMMENDED, EngineConfig, config_dict, run_method
from .report import write_all
from .teacher import (PRICES, CallCost, CapExceeded, Ledger, LLMTeacher, SimulatedTeacher, approx_prompt_tokens,
                      build_system_prompt, precompute)

ROOT = Path.cwd()


def _load_env(path: Path = Path(".env")) -> None:
    """KEY=VALUE lines of ./.env into os.environ (the file is gitignored; it holds API keys)."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")


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
    fewshot = D.load_fewshot(pools, args.shots)
    stream, fewshot = D.build_stream(pools, sc, fewshot)
    system_prompt = build_system_prompt(pools.intents, fewshot)
    if args.teacher == "simulated":
        teacher = SimulatedTeacher(pd.concat([pools.banking, pools.oos]), acc=args.acc, seed=args.seed)
        name = f"simulated (acc {args.acc:.0%})"
    elif args.labels:
        # Shared pool labels (gdistill label-pool): read-only, never calls the API, costs nothing.
        teacher = LLMTeacher(args.teacher, args.model, system_prompt, list(pools.intents) + [D.OOS],
                             cache_path=args.labels, readonly=True)
        name = f"{args.teacher}:{args.model}"
    else:
        if not args.model:
            sys.exit("--model is required for a real teacher")
        n_calls = len(stream)
        sys_tok = approx_prompt_tokens(system_prompt)
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


def cmd_label_pool(args) -> None:
    """Label every unique Banking77 + CLINC out-of-scope text once, into a shared cache."""
    if not args.model:
        sys.exit("--model is required")
    price = PRICES[args.price]
    paid = price.input > 0 or price.output > 0
    if paid and args.cap_usd is None:
        sys.exit("a paid teacher needs --cap-usd (hard spend cap)")
    pools = D.load_pools(args.raw)
    fewshot = D.load_fewshot(pools, args.shots)
    system_prompt = build_system_prompt(pools.intents, fewshot)
    pool = D.pool_texts(pools, fewshot)
    if args.limit:
        pool = pool.head(args.limit)
    labels_path = Path(args.labels or f"labels/{args.model.replace('/', '_')}.jsonl")
    ledger = Ledger(args.ledger, args.cap_usd)
    teacher = LLMTeacher(args.teacher, args.model, system_prompt, list(pools.intents) + [D.OOS],
                         cache_path=labels_path, base_url=args.base_url, max_workers=args.workers,
                         price=price, ledger=ledger)
    cached = teacher._load_cache()
    todo = sum(teacher._key(t) not in cached for t in pool.text)
    sys_tok = approx_prompt_tokens(system_prompt)
    est_hi = todo * ((sys_tok + 25) * price.input + 8 * price.output) / 1e6
    print(f"{len(pool):,} texts in this pool run, {todo:,} not yet labelled; prompt ~{sys_tok:,} tokens (rough).")
    print(f"Upper bound without prompt caching ({price.name}): ${est_hi:.2f}. "
          f"Spent so far on {teacher.name}: ${ledger.spent(teacher.name):.4f}; cap {args.cap_usd}.")
    if todo and not args.yes:
        sys.exit("Add --yes to make the calls. Answers are cached in the labels file; reruns are free.")
    stop = None
    t0 = time.time()
    try:
        teacher.label_all(pool.text.tolist())
    except CapExceeded as e:
        stop = str(e)
    cached = teacher._load_cache()
    recs = [cached[teacher._key(t)] for t in pool.text if teacher._key(t) in cached]
    gold = dict(zip(pool.text, pool.gold))
    if recs:
        r = pd.DataFrame(recs)
        r["gold"] = r.text.map(gold)
        ok = (r.label == r.gold)
        print(f"\n{len(r):,} labelled in {time.time() - t0:.0f}s with {teacher.name}")
        print(f"  accuracy vs human labels: {ok.mean():.1%}   unparseable replies: {(r.label == 'unparseable').mean():.1%}")
        print(f"  per call: input {r.input_tokens.mean():.1f}, cache-read {r.cache_read_tokens.mean():.1f}, "
              f"cache-write {r.cache_write_tokens.mean():.1f}, output {r.output_tokens.mean():.1f} tokens")
        print(f"  calls with a cache read: {(r.cache_read_tokens > 0).mean():.1%}")
        if "usd" in r:
            print(f"  real cost per call: ${r.usd.mean():.6f}  (these {len(r):,} calls: ${r.usd.sum():.4f})")
    print(f"  ledger: ${ledger.spent(teacher.name):.4f} spent on {teacher.name} (cap {args.cap_usd})")
    if stop:
        sys.exit(f"STOPPED: {stop}")


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
    if args.preset == "recommended":      # student and alpha_intent of the chosen operating point; the map stays TF-IDF
        overrides = {"alpha_intent": RECOMMENDED["alpha_intent"], "balance": RECOMMENDED["balance"], **overrides}
        args.student = args.student or RECOMMENDED["student_backend"]
    student = args.student or args.map     # --map sbert also gives the sbert student, unless --student says otherwise
    cfg = EngineConfig(seed=meta["seed"], map_backend=args.map, student_backend=student, **overrides)
    if "sbert" in (args.map, student):
        from . import embed
        t0 = time.time()
        embed.encode(stream.text.tolist())
        print(f"encoded {stream.text.nunique():,} texts with {embed.MODEL} in {time.time() - t0:.0f}s")
    out = run / "results"
    out.mkdir(exist_ok=True)
    for name in args.methods.split(","):
        t0 = time.time()
        res = run_method(stream, tdf, pools.intents, METHODS[name], cfg, cost)
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
                results[name] = load_pickle(f)
    if not any(m in results for m in ("static", "retrain", "global", "cortisol")):
        sys.exit("run `gdistill replay` first (the report needs at least one gated method)")
    op = ""
    if getattr(args, "operating_points", None):
        from .report import operating_points_section
        g = json.loads(Path(getattr(args, "grid", "docs/grid.json")).read_text())
        d = g["default"]
        names = {"retrain": "static gate + retraining", "global": "global cortisol + retraining", "cortisol": "the cortisol field"}
        student = "MiniLM" if d["student"] == "minilm" else "TF-IDF"
        cap = (f"Default (ringed): <b>{names[d['method']]}</b>, {student} student, alpha_intent {d['alpha']:g}. It is the setting with the "
               f"fewest LLM calls whose overall accuracy stays within {-g['bar_pts']:g} points of LLM-only and whose launch-question accuracy "
               f"stays within {-g['launch_bar_pts']:g} points of it, on every one of seeds {', '.join(map(str, g['seeds']))}. "
               f"Settings outside either bar are still shown; hover for overall accuracy and cheap-path errors. ")
        if RECOMMENDED.get("balance"):
            cap += ("This page's run adds class-balanced retraining (a new intent is weighted up to its nearest old neighbour until it has "
                    "50 labels), which was found after the grid was run; the grid points above are without it. ")
        if g["field_on_curve"]:
            cap += "The cortisol field is on the curve at one point only and is not the default, so it stays experimental."
        else:
            cap += "The cortisol field is not on the trade-off curve, so it is experimental."
        op = operating_points_section(Path(args.operating_points).read_text(encoding="utf-8"), cap)
    write_all(results, meta, run, focus=getattr(args, "focus", "cortisol"), operating_points=op)
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
    p = argparse.ArgumentParser(prog="gdistill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    q.add_argument("--labels", default=None, help="shared pool labels (from label-pool): read them, never call")

    q = sub.add_parser("label-pool", help="label every unique pool text once into a shared cache, with a spend cap")
    q.add_argument("--teacher", default="anthropic", choices=["anthropic", "openai"])
    q.add_argument("--model", default=None)
    q.add_argument("--base-url", default=None)
    q.add_argument("--price", default="haiku-5.5", choices=sorted(PRICES))
    q.add_argument("--labels", default=None, help="output cache (default labels/<model>.jsonl)")
    q.add_argument("--ledger", default="labels/spend.json", help="spend ledger shared across runs")
    q.add_argument("--cap-usd", type=float, default=None, help="hard cap on total spend for this teacher")
    q.add_argument("--limit", type=int, default=0, help="label only the first N pool texts (pilot)")
    q.add_argument("--shots", type=int, default=1)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--yes", action="store_true", help="really make the (paid) calls")

    def replay_args(q):
        q.add_argument("--methods", default="llm_only,static,retrain,global,cortisol")
        q.add_argument("--map", default="tfidf", choices=["tfidf", "sbert"], help="map space backend")
        q.add_argument("--preset", default=None, choices=["recommended"],
                       help="apply the recommended operating point (MiniLM student; alpha_intent defaults to 0.03); report it with --focus retrain")
        q.add_argument("--student", default=None, choices=["tfidf", "sbert"],
                       help="student backend (default: same as --map; sbert = frozen MiniLM + logistic regression)")
        q.add_argument("--config", default=None, help='JSON overrides for EngineConfig, e.g. \'{"alpha_intent":0.02}\'')

    q = sub.add_parser("replay", help="replay the stream for each method")
    q.add_argument("--run", default="runs/simulated-s0")
    q.add_argument("--price", default=None, choices=sorted(PRICES))
    replay_args(q)

    q = sub.add_parser("report", help="REPORT.md, results.json and replay.html for a run")
    q.add_argument("--run", default="runs/simulated-s0")
    q.add_argument("--focus", default="cortisol", choices=["static", "retrain", "global", "cortisol"],
                   help="method the dashboard headline describes")
    q.add_argument("--operating-points", default=None, help="SVG from scripts/operating_points.py to add to the page")
    q.add_argument("--grid", default="docs/grid.json", help="grid summary used for the operating-point caption")

    q = sub.add_parser("demo", help="data + simulated labels + replay + report")
    stream_args(q)
    replay_args(q)

    args = p.parse_args(argv)
    _load_env()
    {"data": cmd_data, "label": cmd_label, "label-pool": cmd_label_pool, "replay": cmd_replay, "report": cmd_report, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    main()
