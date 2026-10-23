"""gdistill bundle | serve | traffic."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd


def cmd_bundle(args) -> None:
    """Train students on the shadow-mode history of a seed's stream (teacher labels from the cache)."""
    from .. import data as D
    from ..teacher import LLMTeacher, build_system_prompt
    from .bundle import build_bundle
    pools = D.load_pools(args.raw)
    fewshot = D.load_fewshot(pools, 1)
    stream, _ = D.build_stream(pools, D.StreamConfig(seed=args.seed), fewshot)
    hist = stream[stream.phase == "history"]
    t = LLMTeacher("anthropic", "claude-haiku-5-5", build_system_prompt(pools.intents, fewshot), list(pools.intents) + [D.OOS],
                   cache_path="labels/claude-haiku-5-5.jsonl", readonly=True)
    recs = t.label_all(hist.text.tolist())
    labels = [r["label"] for r in recs]
    version = args.version or Path(args.out).name
    b = build_bundle(hist.text.tolist(), labels, list(pools.intents), version=version, seed=args.seed,
                     meta={"source": f"history of seed {args.seed} stream, Haiku labels", "n_history": len(hist)})
    b.save(args.out)
    print(f"{version}: {len(hist)} history queries, intent holdout acc {b.intent.holdout_acc:.1%}, "
          f"starting levels scope {b.q0_scope:.2f} intent {b.q0_intent:.2f} -> {args.out}")


def cmd_serve(args) -> None:
    import uvicorn
    from .app import make_app
    app = make_app(args.bundle)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_traffic(args) -> None:
    import httpx
    stream = pd.read_csv(Path(args.stream) / "stream.csv", keep_default_na=False)
    stream = stream[stream.phase != "history"]
    if args.limit:
        stream = stream.head(args.limit)
    gap = 1.0 / args.rate
    n = esc = 0
    t_start = time.time()
    with httpx.Client(timeout=120) as c:
        for r in stream.itertuples():
            t0 = time.time()
            try:
                out = c.post(args.url.rstrip("/") + "/triage", json={"text": r.text}).json()
            except Exception as e:                      # keep going: a sleeping free host wakes slowly
                print(f"request failed: {e}")
                time.sleep(2)
                continue
            n += 1
            esc += out.get("decision") == "escalate"
            if n % 50 == 0:
                print(f"{n} sent, {esc / n:.0%} escalated, {n / (time.time() - t_start):.1f} req/s")
            time.sleep(max(0.0, gap - (time.time() - t0)))
    m = httpx.get(args.url.rstrip("/") + "/metrics").json()
    print(json.dumps({k: m[k] for k in ("model_version", "decisions_total", "escalation_rate", "estimated_leak_rate", "leak_budget", "spend")}, indent=1))
