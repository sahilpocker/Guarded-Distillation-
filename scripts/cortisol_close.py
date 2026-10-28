"""Closing the cortisol question with no API spend: E3 (a policy change) and E5 (ablations of the full field).

    python scripts/cortisol_close.py run  --stream policy|plain --tag NAME --variants a,b,c [--forget] [--seeds 3 4 5 6 7 8]
    python scripts/cortisol_close.py table --stream policy|plain --tag NAME --ref NAME2

Streams and cached Haiku labels come from runs/grid-minilm-a03-s{k}. `policy` applies the E3 change (docs/ROADMAP.md): from
40% of the live stream on (query t >= 3200), every teacher label `card_arrival` becomes `card_delivery_estimate` and the gold
label is changed the same way. Engine settings are the recommended ones (MiniLM student, alpha_intent 0.03, balanced
retraining) with the original margin router, which is what the cortisol methods were built and compared with.
Per-query results go to runs/cortisol_close/{tag}-{stream}-s{k}.pkl (arrays); tables use a paired bootstrap over queries.
"""
import argparse
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from guarded_distillation import data as D, embed  # noqa: E402
from guarded_distillation.cli import _load_run  # noqa: E402
from guarded_distillation.engine import METHODS, EngineConfig, run_method  # noqa: E402
from guarded_distillation.teacher import PRICES, CallCost  # noqa: E402
from boot import paired  # noqa: E402

OUT = Path("runs/cortisol_close")
CUT, SRC, DST = 3200, "card_arrival", "card_delivery_estimate"
FULL = METHODS["cortisol"]


def variants() -> dict:
    v = {name: METHODS[name] for name in ("llm_only", "static", "retrain", "global", "cortisol", "semantic")}
    v["no_novelty"] = dataclasses.replace(FULL, name="no_novelty", novelty=False)
    v["no_audits"] = dataclasses.replace(FULL, name="no_audits", directed_audits=False)
    v["no_local"] = dataclasses.replace(FULL, name="no_local", local=False)
    v["no_hotcell"] = dataclasses.replace(FULL, name="no_hotcell", hot_cell_retrain=False)
    return v


def policy_change(stream, tdf):
    stream, tdf = stream.copy(), tdf.copy()
    after = stream.t.to_numpy() >= CUT
    stream.loc[after & (stream.gold == SRC), "gold"] = DST
    tdf.loc[after & (tdf.teacher == SRC).to_numpy(), "teacher"] = DST
    return stream, tdf


def arrays(r, ref, stream_live_gold_changed):
    c = r.calls[r.calls.phase != "history"].reset_index(drop=True)
    t = c.t.to_numpy()
    aff = stream_live_gold_changed
    first = np.zeros(len(c), bool)
    first[np.flatnonzero((c.phase == "launch").to_numpy())[:600]] = True
    return {
        "correct": c.correct.to_numpy(), "llm": c.llm_call.to_numpy(), "wrong_cheap": ((c.decision == "accept") & ~c.correct).to_numpy(),
        "post": t >= CUT, "aff": aff, "launch": c.is_launch.to_numpy(), "first600": first & c.is_launch.to_numpy(),
        "cost": c.cost_usd.to_numpy(), "forgot": sum(1 for e in r.events if e["kind"] == "forgot"),
        "promoted": sum(1 for e in r.events if e["kind"] == "promoted"),
    }


def run_seed(k, stream_kind, names, forget, tag, alpha=0.03):
    run = Path(f"runs/grid-minilm-a03-s{k}")
    stream, tdf, meta = _load_run(run)
    pools = D.load_pools("data/raw")
    price = PRICES[meta.get("price", "haiku-5.5")]
    cost = np.array([CallCost(int(r.input_tokens), int(r.output_tokens), int(r.cache_read_tokens), int(r.cache_write_tokens)).usd(price)
                     for r in tdf.itertuples()])
    changed = np.zeros(len(stream), bool)
    if stream_kind == "policy":
        changed = ((stream.t.to_numpy() >= CUT) & (stream.gold == SRC)).to_numpy()
        stream, tdf = policy_change(stream, tdf)
    live = (stream.phase != "history").to_numpy()
    cfg = EngineConfig(seed=meta["seed"], map_backend="tfidf", student_backend="sbert", alpha_intent=alpha, balance=True, forget=forget)
    embed.encode(stream.text.tolist())
    V = variants()
    res = {n: run_method(stream, tdf, tuple(meta.get("intents") or pools.intents), V[n], cfg, cost) for n in names}
    out = {n: arrays(r, res.get("llm_only"), changed[live]) for n, r in res.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}-{stream_kind}-s{k}.pkl", "wb") as f:
        pickle.dump(out, f)
    return k


def cmd_run(a):
    from concurrent.futures import ProcessPoolExecutor
    names = a.variants.split(",")
    if "llm_only" not in names:
        names = ["llm_only"] + names
    with ProcessPoolExecutor(min(len(a.seeds), 6)) as ex:
        for k in ex.map(run_seed, a.seeds, [a.stream] * len(a.seeds), [names] * len(a.seeds), [a.forget] * len(a.seeds), [a.tag] * len(a.seeds), [a.alpha] * len(a.seeds)):
            print("seed", k, "done", flush=True)


def load(tag, stream_kind, seeds):
    return [pickle.load(open(OUT / f"{tag}-{stream_kind}-s{k}.pkl", "rb")) for k in seeds]


def cell(per, A, B, key, mask):
    """mean difference A-B in points with a paired bootstrap interval, or None when the mask is empty"""
    a = [dict(x[A], d=x[A][key].astype(float)) for x in per]
    b = [dict(x[B], d=x[B][key].astype(float)) for x in per]
    return paired(a, b, "d", mask_key=mask)


def cmd_table(a):
    per = load(a.tag, a.stream, a.seeds)
    names = [n for n in per[0] if n != "llm_only"]
    rows = []
    print(f"{'variant':<12}{'calls %':>9}{'wrong-cheap/1k':>16}{'acc vs LLM (pts)':>26}{'affected-intent acc vs LLM':>34}{'first600 launch vs LLM':>30}")
    for n in names:
        calls = np.mean([x[n]["llm"].mean() for x in per]) * 100
        wc = np.mean([x[n]["wrong_cheap"].mean() for x in per]) * 1000
        acc = cell(per, n, "llm_only", "correct", None)
        aff = cell(per, n, "llm_only", "correct", ["aff"]) if a.stream == "policy" else None
        f6 = cell(per, n, "llm_only", "correct", ["first600"])
        fm = lambda r: "n/a" if r is None else f"{r['mean']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]"
        print(f"{n:<12}{calls:>9.1f}{wc:>16.1f}{fm(acc):>26}{fm(aff):>34}{fm(f6):>30}")
        rows.append({"variant": n, "calls": calls, "wrong_cheap_per_1k": wc, "acc": acc, "aff": aff, "first600": f6,
                     "promoted": float(np.mean([x[n]["promoted"] for x in per])), "forgot": float(np.mean([x[n]["forgot"] for x in per]))})
    if a.ref:
        print(f"\npaired differences vs {a.ref} (variant minus {a.ref}), points; wrong-cheap and calls in points of queries")
        for n in names:
            if n == a.ref:
                continue
            d = {k: cell(per, n, a.ref, k, mk) for k, mk in (("correct", None), ("llm", None), ("wrong_cheap", None))}
            if a.stream == "policy":
                d["aff"] = cell(per, n, a.ref, "correct", ["aff"])
            print(f"{n:<12}" + "  ".join(f"{k}={v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] ({v['wins']}/{v['n_seeds']})" for k, v in d.items()))
            rows[[r['variant'] for r in rows].index(n)]["vs_" + a.ref] = d
    n_aff = [int(x[names[0]]["aff"].sum()) for x in per]
    print("affected queries per seed:", n_aff)
    (OUT / f"{a.tag}-{a.stream}-table.json").write_text(json.dumps(rows, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--stream", choices=["policy", "plain"], default="policy")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variants", default="static,retrain,global,cortisol,semantic")
    ap.add_argument("--forget", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.03, help="alpha_intent (a stricter gate escalates more: used to match call budgets)")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8])
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_table(a)


if __name__ == "__main__":
    main()
