"""Semantic cortisol vs the balanced plain gate, on the cached Haiku labels (no API calls unless --sensor).

    python scripts/semantic.py --seeds 0 1 2 --tag base --methods llm_only,retrain,semantic [--config '{...}']
    python scripts/semantic.py --seeds 0 1 2 --tag sens --methods semantic_llm --sensor

Streams and teacher labels come from runs/grid-minilm-a03-s{k} (MiniLM student, TF-IDF map). The
engine settings are the recommended ones (alpha_intent 0.03, MiniLM student) plus class-balanced
retraining for every method. Writes runs/semantic/{tag}-s{k}.json and prints a table.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from cortisol import data as D, embed  # noqa: E402
from cortisol.cli import _load_env, _load_run  # noqa: E402
from cortisol.engine import METHODS, EngineConfig, run_method  # noqa: E402
from cortisol.sensor import HaikuSensor  # noqa: E402
from cortisol.teacher import PRICES, CallCost  # noqa: E402

FIRST = 600     # "first 600 launch-phase queries"


def metrics(r, ref):
    c = r.calls[r.calls.phase != "history"]
    ph = c[c.phase == "launch"].head(FIRST)
    ref_c = ref.calls[ref.calls.phase != "history"]
    ref_ph = ref_c[ref_c.phase == "launch"].head(FIRST)
    acc = float(c.correct.mean())
    return {
        "calls": float(c.llm_call.mean() * 100),
        "acc": acc * 100,
        "acc_delta": (acc - float(ref_c.correct.mean())) * 100,
        "launch": float(c[c.is_launch].correct.mean() * 100),
        "launch_delta": float((c[c.is_launch].correct.mean() - ref_c[ref_c.is_launch].correct.mean()) * 100),
        "launch600": float(ph[ph.is_launch].correct.mean() * 100),
        "launch600_delta": float((ph[ph.is_launch].correct.mean() - ref_ph[ref_ph.is_launch].correct.mean()) * 100),
        "errors_per_1k": float(((c.decision == "accept") & ~c.correct).mean() * 1000),
        "extra_calls_pct": float(c.extra_calls.sum() / len(c) * 100) if "extra_calls" in c else 0.0,
        "probation_pct": float(c.probation.mean() * 100) if "probation" in c else 0.0,
        "cost_per_1k": float(c.cost_usd.sum() / len(c) * 1000),
        "n_launch600": int(ph.is_launch.sum()),
    }


def run_seed(k, methods, cfg_over, sensor, tag, runs="grid-minilm-a03-s"):
    run = Path(f"runs/{runs}{k}")
    stream, tdf, meta = _load_run(run)
    pools = D.load_pools("data/raw")
    price = PRICES[meta.get("price", "haiku-5.5")]
    cost = np.array([CallCost(int(r.input_tokens), int(r.output_tokens), int(r.cache_read_tokens),
                              int(r.cache_write_tokens)).usd(price) for r in tdf.itertuples()])
    cfg = EngineConfig(**({"seed": meta["seed"], "map_backend": "tfidf", "student_backend": "sbert", "alpha_intent": 0.03, "balance": True} | cfg_over))
    embed.encode(stream.text.tolist())
    res = {}
    for name in methods:
        res[name] = run_method(stream, tdf, tuple(meta.get("intents") or pools.intents), METHODS[name], cfg, cost, sensor=sensor if METHODS[name].sensor else None)
    ref = res["llm_only"]
    out = {m: metrics(r, ref) for m, r in res.items() if m != "llm_only"}
    out["_events"] = {m: [e for e in r.events if e["kind"] in ("sensor", "released")][:40] for m, r in res.items() if m != "llm_only"}
    Path("runs/semantic").mkdir(parents=True, exist_ok=True)
    Path(f"runs/semantic/{tag}-s{k}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(f"runs/semantic/{tag}-s{k}.pkl", "wb") as f:
        pickle.dump(res, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--methods", default="llm_only,retrain,semantic")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--runs", default="grid-minilm-a03-s", help="run folder prefix; the seed is appended (e.g. clinc-travel-s)")
    ap.add_argument("--sensor", action="store_true", help="allow Claude calls for the LLM sensor (cap $0.50)")
    a = ap.parse_args()
    methods = a.methods.split(",")
    if "llm_only" not in methods:
        methods = ["llm_only"] + methods
    sensor = None
    if a.sensor:
        _load_env()
        sensor = HaikuSensor()
    from concurrent.futures import ProcessPoolExecutor
    over = json.loads(a.config)
    if sensor is None:
        with ProcessPoolExecutor(len(a.seeds)) as ex:
            outs = list(ex.map(run_seed, a.seeds, [methods] * len(a.seeds), [over] * len(a.seeds), [None] * len(a.seeds), [a.tag] * len(a.seeds), [a.runs] * len(a.seeds)))
    else:      # sequential: one ledger, one cache file
        outs = [run_seed(k, methods, over, sensor, a.tag, a.runs) for k in a.seeds]
        print(f"sensor: {sensor.calls} new calls, ${sensor.spent():.4f} spent in total (cap $0.50)")
    keys = [k for k in outs[0] if not k.startswith("_")]
    cols = ["calls", "acc_delta", "launch_delta", "launch600_delta", "errors_per_1k", "extra_calls_pct"]
    print(f"{'method':<14}" + "".join(f"{c:>17}" for c in cols))
    for m in keys:
        print(f"{m:<14}" + "".join(f"{np.mean([o[m][c] for o in outs]):>9.2f} {min(o[m][c] for o in outs):>+6.1f}" if False else
                                    f"{np.mean([o[m][c] for o in outs]):>17.2f}" for c in cols))
        for k, o in zip(a.seeds, outs):
            print(f"   seed {k}: " + "  ".join(f"{c}={o[m][c]:.2f}" for c in cols))


if __name__ == "__main__":
    main()
