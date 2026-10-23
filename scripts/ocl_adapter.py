"""Run Online Cascade Learning (Nie et al., ICML 2024; github.com/flitternie/online_cascade_learning) on our streams.

    python scripts/ocl_adapter.py prepare --family banking --seed 0
    python scripts/ocl_adapter.py run --family banking --seed 0 --mu 0.05      # needs OCL's dependencies (torch, datasets, litellm)

Their code is used unmodified: a cascade of one logistic regression (TF-IDF + SGD, their `sklearn` model) in front of
the LLM, with their learned deferral network and imitation-learning updates. Differences from our setting, all forced
by their design: no shadow-mode history (it learns from the LLM from the first query, so the 4,000 history queries are
in the stream and only the 8,000 live queries are scored), the LLM labels come from our Haiku cache as a pre-generated
file, their TF-IDF vocabulary is fitted on the stream's text, and there is no retraining schedule, no audits and no leak
budget: deferral is learned from the cost coefficient mu alone. Run with the environment in third_party/ocl-env.
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

OCL = Path(os.environ.get("OCL_DIR", "third_party/online_cascade_learning")).resolve()
FAMILIES = {"banking": "grid-minilm-a03-s", "credit_cards": "clinc-credit_cards-s", "travel": "clinc-travel-s"}
OUT = Path("runs/ocl").resolve()
OOS = "out_of_scope"


def labels_for(meta_path: Path) -> list[str]:
    meta = json.loads(meta_path.read_text())
    intents = meta.get("intents")
    if not intents:                                     # Banking77 runs: the 77 intents are the ones in the stream
        intents = None
    return intents


def prepare(fam: str, seed: int) -> Path:
    run = Path(__file__).resolve().parent.parent / "runs" / f"{FAMILIES[fam]}{seed}"
    stream = pd.read_csv(run / "stream.csv", keep_default_na=False)
    tdf = pd.read_csv(run / "teacher.csv", keep_default_na=False)
    intents = labels_for(run / "meta.json")
    if intents is None:
        intents = sorted(set(stream.gold) | set(tdf.teacher))
        intents = [i for i in intents if i not in (OOS, "unparseable")]
    labels = list(intents) + [OOS]
    idx = {l: i for i, l in enumerate(labels)}
    d = OUT / f"{fam}-s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    gold = stream.gold.map(idx)
    llm = tdf.teacher.map(lambda t: idx.get(t, len(labels) - 1))      # an unparseable reply counts as out_of_scope
    pd.DataFrame({"text": stream.text, "label": gold}).to_csv(d / "stream.csv", index=False)
    (d / "llm.txt").write_text("\n".join(str(int(v)) for v in llm) + "\n")
    meta = {"num_labels": len(labels), "phase": stream.phase.tolist(), "is_launch": stream.is_launch.astype(str).str.lower().eq("true").tolist()}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def run(fam: str, seed: int, mu: float) -> dict:
    d = prepare(fam, seed)
    meta = json.loads((d / "meta.json").read_text())
    sys.path.insert(0, str(OCL))
    os.chdir(OCL)
    import datasets
    import torch
    from cascade import OnlineCascade
    from data import CustomDataset
    from models import ModelWrapper, llm_factory, model_factory
    from utils import load_config, set_seed

    mod = types.ModuleType("gdstream")                    # the data module OCL expects: labels are already integers
    mod.DATASET = "gdstream"
    mod.preprocess = lambda data: data
    mod.postprocess = lambda s: int(s)
    sys.modules["gdstream"] = mod
    absd = d
    cfg_path = absd / f"config-{mu}.yaml"
    cfg_path.write_text(f"""
log: {absd}/logs/
seed: {seed}
data: {{name: gdstream, env: gdstream, path: {absd}/stream.csv, num_labels: {meta['num_labels']}}}
llm: {{name: haiku, source: {absd}/llm.txt}}
mu: [{mu}]
models:
  - name: LR
    source: sklearn.linear_model
    model: SGDClassifier
    model_args: {{loss: log_loss}}
    cache_size: 8
    cost: 1
    wrapper: {{learning_rate: 0.0007, regularization: 0.0001, decaying_factor: 0.99, calibration: 0.4}}
""")
    config = load_config(str(cfg_path))
    set_seed(seed)
    data = datasets.Dataset.from_pandas(pd.read_csv(config.data.path))
    llm = llm_factory(config.llm, config.data)
    data = CustomDataset.create(data, llm)             # no shuffle and no split: the stream keeps its order
    config.data.corpus = data["text"]
    wrappers = []
    for mc in config.models:
        mc.device = "cpu"
        m = model_factory(mc, config.data)
        w = ModelWrapper(m, mc)
        w.to(w.device)
        wrappers.append(w)
    cascade = OnlineCascade(wrappers, mu, config=config)
    cascade.run(data, mod)
    log = sorted((absd / "logs" / "gdstream").glob(f"LR_haiku_{mu:.8f}.log"))[-1]
    rows = [l.strip().split(",") for l in log.read_text().splitlines()]
    pred = np.array([int(r[0]) for r in rows])
    acted = np.array([int(r[-1]) for r in rows])             # 0 = the LR answered, 1 = the LLM
    gold = pd.read_csv(absd / "stream.csv").label.to_numpy()
    llm_l = np.array([int(x) for x in (absd / "llm.txt").read_text().split()])
    final = np.where(acted == 1, llm_l, pred)
    ph = np.array(meta["phase"])
    live = ph != "history"
    il = np.array(meta["is_launch"])
    lp = np.flatnonzero(ph == "launch")[:600]
    ref_final = llm_l
    out = {"family": fam, "seed": seed, "mu": mu, "calls": float(acted[live].mean() * 100),
           "acc": float((final == gold)[live].mean() * 100), "llm_acc": float((llm_l == gold)[live].mean() * 100),
           "launch": float((final == gold)[live & il].mean() * 100), "llm_launch": float((llm_l == gold)[live & il].mean() * 100),
           "launch600": float((final == gold)[lp][il[lp]].mean() * 100), "llm_launch600": float((llm_l == gold)[lp][il[lp]].mean() * 100),
           "leak_rate_vs_teacher": float(((acted == 0) & (final != llm_l))[live].mean())}
    (OUT / f"{fam}-s{seed}-mu{mu}.json").write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("--family", default="banking")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd == "prepare":
        print(prepare(a.family, a.seed))
    else:
        print(json.dumps(run(a.family, a.seed, a.mu), indent=1))
