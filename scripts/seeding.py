"""Synthetic seeding for new intents (see docs/RESULTS.md, "Synthetic seeding").

    python scripts/seeding.py generate           # ~20 synthetic examples per new intent (NVIDIA Nemotron), from the name,
                                                 # a one-line description and the fixed few-shot example ONLY
    python scripts/seeding.py streams --seeds 0 1 2 3 4 5 6 7 8     # held-out-intent streams (+ CLINC seeds 6-8), cached labels
    python scripts/seeding.py grid | select | final | report        # experiment (below)

When a new intent is declared (the first query of the launch phase) its synthetic examples join the training set at once
(balanced), the student is retrained, and the intent starts on probation. Real stream queries are never used to make seeds.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

DESC = json.loads(Path("data/synth/intent_descriptions.json").read_text(encoding="utf-8"))
NEAR, FAR = DESC["near"], DESC["far"]
GEN_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
SEEDS_FILE = Path("data/synth/seeds.json")


def fewshot_text(intent: str) -> str:
    """The one fixed few-shot example of the intent (the one in the teacher prompt), never a stream query."""
    if ":" in intent:
        import clinc_domain
        _, fewshot, *_ = clinc_domain.setup(intent.split(":")[0])
    else:
        from guarded_distillation import data as D
        fewshot = D.load_fewshot(D.load_pools("data/raw"), 1)
    return fewshot[fewshot.label == intent].text.iloc[0]


def cmd_generate():
    from openai import OpenAI, RateLimitError
    from guarded_distillation.cli import _load_env
    from concurrent.futures import ThreadPoolExecutor
    _load_env()
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"].strip(), timeout=120, max_retries=0)
    intents = NEAR + FAR + [k for k in DESC["descriptions"] if ":" in k]
    out = json.loads(SEEDS_FILE.read_text()) if SEEDS_FILE.exists() else {}
    todo = [i for i in intents if i not in out]
    shots = {i: fewshot_text(i) for i in todo}

    def run(intent):
        name = intent.split(":")[-1].replace("_", " ")
        prompt = (f"Intent name: {name}\nDescription: {DESC['descriptions'][intent]}\nOne example message: \"{shots[intent]}\"\n\n"
                  "Write 20 different messages a customer might send for this intent. Vary the length, tone and wording: some short, some long, a few "
                  "casual with typos. Do not repeat the example. Reply with a JSON list of 20 strings and nothing else.")
        for wait in (2, 5, 10, 20, 40, 60):
            try:
                r = client.chat.completions.create(model=GEN_MODEL, temperature=0.8, max_tokens=1200,
                                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                                   messages=[{"role": "system", "content": "You write realistic customer messages to test a classifier. Reply with JSON only."},
                                                             {"role": "user", "content": prompt}])
                break
            except RateLimitError:
                time.sleep(wait)
        else:
            raise RuntimeError(f"rate limited on {intent}")
        raw = r.choices[0].message.content or ""
        m = re.search(r"\[.*\]", raw, re.S)
        items = json.loads(m.group(0)) if m else []
        seen, keep = {shots[intent].lower()}, []
        for t in items:
            if isinstance(t, str) and t.strip() and t.strip().lower() not in seen:
                seen.add(t.strip().lower())
                keep.append(t.strip())
        return intent, keep[:20]

    with ThreadPoolExecutor(2) as ex:
        for intent, keep in ex.map(run, todo):
            out[intent] = keep
            print(f"{intent:<42} {len(keep)} examples", flush=True)
            SEEDS_FILE.write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_streams(seeds):
    from guarded_distillation import data as D
    from guarded_distillation.teacher import LLMTeacher, build_system_prompt
    import clinc_domain
    pools = D.load_pools("data/raw")
    fewshot = D.load_fewshot(pools, 1)
    t = LLMTeacher("anthropic", "claude-haiku-5-5", build_system_prompt(pools.intents, fewshot), list(pools.intents) + [D.OOS],
                   cache_path="labels/claude-haiku-5-5.jsonl", readonly=True)
    kinds = {**{i: "near" for i in NEAR}, **{i: "far" for i in FAR}}
    for seed in seeds:
        stream, _ = D.build_stream(pools, D.StreamConfig(seed=seed), fewshot, launch_intents=tuple(NEAR + FAR), launch_kinds=kinds)
        recs = t.label_all(stream.text.tolist(), progress=False)
        tdf = pd.DataFrame([{k: r[k] for k in ("label", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")} for r in recs]).rename(columns={"label": "teacher"})
        tdf.insert(0, "t", stream.t.values)
        run = Path(f"runs/heldout-s{seed}")
        run.mkdir(parents=True, exist_ok=True)
        stream.to_csv(run / "stream.csv", index=False)
        tdf.to_csv(run / "teacher.csv", index=False)
        live = (stream.phase != "history").to_numpy()
        meta = {"run": run.name, "teacher": "anthropic:claude-haiku-5-5", "seed": seed, "history": 4000, "live": 8000, "price": "haiku-5.5",
                "teacher_accuracy_live": float((tdf.teacher.to_numpy() == stream.gold.to_numpy())[live].mean())}
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
        k = stream[live & stream.is_launch.to_numpy()].kind.value_counts().to_dict()
        print(f"{run}: launch queries {k}, teacher acc {meta['teacher_accuracy_live']:.1%}")
    for dom in ("credit_cards", "travel"):                       # CLINC streams for the extra seeds
        for seed in seeds:
            if not Path(f"runs/clinc-{dom}-s{seed}/stream.csv").exists():
                import argparse
                clinc_domain.cmd_streams(argparse.Namespace(domain=dom, seeds=[seed]))


# ----------------------------------------------------------------------------- experiment
FAMILIES = {"heldout": "heldout-s", "credit_cards": "clinc-credit_cards-s", "travel": "clinc-travel-s"}
ARMS = ["none", "declare", "seeded", "none_np", "seeded_np"]
# none: no seeding | declare: declared, probation only | seeded: declared with synthetic examples  (all three: semantic cortisol, probation on)
# *_np: the same without probation (static gate + learned router), where seeding has no safety net to lean on
ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
BUDGETS = [50, 60]
SEM = {"sem_gain": 0.7, "sem_hot": 0.4, "sem_qfit": 0.5}
OUT = Path("runs/seeding")


def rpath(fam, arm, alpha, seed):
    return OUT / f"{fam}-{arm}-a{alpha:g}-s{seed}.json"


def seeds_for(fam: str, stream_texts: set[str]) -> dict[str, list[str]]:
    allseeds = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
    keys = (NEAR + FAR) if fam == "heldout" else [k for k in allseeds if k.startswith(fam + ":")]
    low = {t.lower() for t in stream_texts}
    return {k: [t for t in allseeds[k] if t.lower() not in low] for k in keys}      # never a real stream query


def run_one(fam, arm, alpha, seed):
    p = rpath(fam, arm, alpha, seed)
    if p.exists():
        return json.loads(p.read_text())
    from guarded_distillation import data as D, embed
    from guarded_distillation.cli import _load_run
    from guarded_distillation.engine import METHODS, EngineConfig, run_method
    from guarded_distillation.teacher import PRICES, CallCost
    run = Path(f"runs/{FAMILIES[fam]}{seed}")
    stream, tdf, meta = _load_run(run)
    pools = D.load_pools("data/raw")
    price = PRICES[meta.get("price", "haiku-5.5")]
    cost = np.array([CallCost(int(r.input_tokens), int(r.output_tokens), int(r.cache_read_tokens), int(r.cache_write_tokens)).usd(price) for r in tdf.itertuples()])
    seeds = seeds_for(fam, set(stream.text))
    kw = {"seed": meta["seed"], "map_backend": "tfidf", "student_backend": "sbert", "alpha_intent": alpha, "balance": True, "router": "learned", **SEM}
    if arm == "declare":
        kw["seed_examples"] = {k: [] for k in seeds}
    elif arm.startswith("seeded"):
        kw["seed_examples"] = seeds
    method = "retrain" if arm.endswith("_np") else "semantic"
    if method == "retrain":
        for k in SEM:
            kw.pop(k)
    cfg = EngineConfig(**kw)
    embed.encode(stream.text.tolist())
    intents = tuple(meta.get("intents") or pools.intents)
    ref = run_method(stream, tdf, intents, METHODS["llm_only"], cfg, cost)
    res = run_method(stream, tdf, intents, METHODS[method], cfg, cost)
    c = res.calls[res.calls.phase != "history"].reset_index(drop=True)
    r0 = ref.calls[ref.calls.phase != "history"].reset_index(drop=True)
    kind = stream[stream.phase != "history"].reset_index(drop=True).get("kind", pd.Series([""] * len(c)))
    if fam != "heldout":
        kind = pd.Series(["near" if fam == "credit_cards" else "far"] * len(c)).where(c.is_launch, "")
    first = np.zeros(len(c), bool)
    first[np.flatnonzero((c.phase == "launch").to_numpy())[:600]] = True
    il = c.is_launch.to_numpy()
    acc = lambda m: float(m.mean() * 100) if len(m) else float("nan")      # noqa: E731
    out = {"family": fam, "arm": arm, "alpha": alpha, "seed": seed, "calls": float(c.llm_call.mean() * 100), "acc": acc(c.correct),
           "acc_delta": acc(c.correct) - acc(r0.correct), "launch": acc(c.correct[il]), "launch600": acc(c.correct[first & il]),
           "llm_launch600": acc(r0.correct[first & il]), "leak": float(((c.decision == "accept") & (c.final != c.teacher)).mean()),
           "n_launch600": int((first & il).sum()), "calls_launch600": acc(c.llm_call[first & il]),
           "declared": any(e["kind"] == "declared" for e in res.events)}
    for k in ("near", "far"):
        sel = first & il & (kind.to_numpy() == k)
        if sel.any():
            out[f"launch600_{k}"] = acc(c.correct[sel])
            out[f"llm_launch600_{k}"] = acc(r0.correct[sel])
            out[f"n_launch600_{k}"] = int(sel.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{fam}-{arm}-a{alpha:g}-s{seed}.npz", correct=c.correct.to_numpy(), llm_correct=r0.correct.to_numpy(), is_launch=il, first=first,
                        kind=np.array([{"": 0, "near": 1, "far": 2}[k] for k in kind]), llm_call=c.llm_call.to_numpy())
    p.write_text(json.dumps(out))
    return out


def pool_run(jobs, workers=6):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [j for j in jobs if not rpath(*j).exists()]
    print(f"{len(jobs)} runs, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run_one, *j): j for j in todo}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                print("FAILED", futs[f], repr(e)[:200], flush=True)
            if n % 12 == 0:
                print(f"{n}/{len(todo)}", flush=True)


def interp_alpha(pts, target):
    pts = sorted(pts)
    for (a0, c0), (a1, c1) in zip(pts, pts[1:]):
        if (c0 - target) * (c1 - target) <= 0 and c0 != c1:
            t = (c0 - target) / (c0 - c1)
            return float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
    return None


def cmd_grid():
    pool_run([(f, a, al, sd) for f in FAMILIES for a in ARMS for al in ALPHAS for sd in (0, 1, 2)])


def cmd_select():
    sel = {}
    for f in FAMILIES:
        for a in ARMS:
            pts = []
            for al in ALPHAS:
                rows = [json.loads(rpath(f, a, al, sd).read_text()) for sd in (0, 1, 2) if rpath(f, a, al, sd).exists()]
                if len(rows) == 3:
                    pts.append((al, float(np.mean([r["calls"] for r in rows]))))
            for b in BUDGETS:
                sel[f"{f}|{a}|{b}"] = interp_alpha(pts, b)
    (OUT / "select.json").write_text(json.dumps(sel, indent=1))
    print({k: (None if v is None else round(v, 4)) for k, v in sel.items()})


def cmd_final():
    sel = json.loads((OUT / "select.json").read_text())
    pool_run([(k.split("|")[0], k.split("|")[1], round(v, 5), sd) for k, v in sel.items() if v for sd in range(3, 9)])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "generate":
        cmd_generate()
    elif cmd in ("grid", "select", "final"):
        {"grid": cmd_grid, "select": cmd_select, "final": cmd_final}[cmd]()
    elif cmd == "streams":
        cmd_streams([int(x) for x in sys.argv[sys.argv.index("--seeds") + 1:]] if "--seeds" in sys.argv else list(range(9)))
