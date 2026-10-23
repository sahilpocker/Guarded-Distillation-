"""CLINC150 domain holdout: a whole new domain arrives as the launch event.

    python scripts/clinc_domain.py label --domain credit_cards --cap-usd 1.5 --yes   # Claude calls (once per domain)
    python scripts/clinc_domain.py streams --domain credit_cards --seeds 0 1 2 3 4 5   # free: reads the caches

The 15 intents of one CLINC150 domain are renamed "<domain>:<intent>" and held out of history. The
teacher's prompt lists the 77 Banking77 intents plus those 15 (one example each), as if the
prompt were updated at launch. Banking77 and out-of-scope texts keep their earlier labels (made
with the 77-intent prompt); only the domain's own texts are labelled under the longer prompt.
Streams go to runs/clinc-<domain>-s<seed>; replay them with scripts/semantic.py --runs clinc-<domain>.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from guarded_distillation import data as D  # noqa: E402
from guarded_distillation.cli import _load_env  # noqa: E402
from guarded_distillation.teacher import PRICES, CapExceeded, Ledger, LLMTeacher, approx_prompt_tokens, build_system_prompt  # noqa: E402

MODEL = "claude-haiku-5-5"
BASE_LABELS = "labels/claude-haiku-5-5.jsonl"
DOMAINS = {
    "credit_cards": "credit_score report_lost_card credit_limit rewards_balance new_card application_status card_declined "
                    "international_fees apr redeem_rewards credit_limit_change damaged_card replacement_card_duration "
                    "improve_credit_score expiration_date".split(),
    "travel": "book_flight book_hotel car_rental travel_suggestion travel_alert travel_notification carry_on timezone vaccines "
              "translate flight_status international_visa lost_luggage plug_type exchange_rate".split(),
}


def domain_pool(domain: str, pools: D.Pools, raw="data/raw") -> pd.DataFrame:
    j = json.loads((Path(raw) / "clinc150_data_full.json").read_text())
    keep = set(DOMAINS[domain])
    rows = [(t.strip(), f"{domain}:{lab}") for t, lab in j["train"] + j["val"] + j["test"] if lab in keep]
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text")
    banking = set(pools.banking.text) | set(pools.oos.text)
    return df[~df.text.isin(banking)].reset_index(drop=True)


def setup(domain: str, shots: int = 1):
    pools = D.load_pools("data/raw")
    base_fs = D.load_fewshot(pools, shots)
    pool = domain_pool(domain, pools)
    new_fs = pool.sample(frac=1.0, random_state=0).groupby("label").head(shots)
    fewshot = pd.concat([base_fs, new_fs]).reset_index(drop=True)
    pool = pool[~pool.text.isin(set(new_fs.text))].reset_index(drop=True)
    new_intents = sorted(pool.label.unique())
    intents = tuple(sorted(pools.intents) + new_intents)
    prompt = build_system_prompt(intents, fewshot)
    prompt_base = build_system_prompt(pools.intents, base_fs)
    return pools, fewshot, pool, intents, prompt, prompt_base, base_fs


def teachers(domain, prompt, prompt_base, intents, pools, readonly, ledger=None, workers=8):
    price = PRICES["haiku-5.5"]
    dom = LLMTeacher("anthropic", MODEL, prompt, list(intents) + [D.OOS], cache_path=f"labels/{MODEL}-{domain}.jsonl",
                     max_workers=workers, price=price, ledger=ledger, readonly=readonly)
    base = LLMTeacher("anthropic", MODEL, prompt_base, list(pools.intents) + [D.OOS], cache_path=BASE_LABELS, readonly=True)
    return dom, base


def cmd_label(a):
    _load_env()
    pools, fewshot, pool, intents, prompt, prompt_base, _ = setup(a.domain)
    ledger = Ledger("labels/spend.json", a.cap_usd)
    dom, _ = teachers(a.domain, prompt, prompt_base, intents, pools, readonly=False, ledger=ledger, workers=a.workers)
    cached = dom._load_cache()
    todo = sum(dom._key(t) not in cached for t in pool.text)
    est = todo * (approx_prompt_tokens(prompt) + 25) * PRICES["haiku-5.5"].input / 1e6
    print(f"{a.domain}: {len(pool):,} texts, {todo:,} unlabelled; prompt ~{approx_prompt_tokens(prompt):,} tokens; "
          f"upper bound without prompt caching ${est:.3f}; spent so far ${ledger.spent(dom.name):.4f}, cap {a.cap_usd}")
    if todo and not a.yes:
        sys.exit("add --yes to make the calls")
    try:
        recs = dom.label_all(pool.text.tolist())
    except CapExceeded as e:
        sys.exit(f"STOPPED: {e}")
    r = pd.DataFrame(recs)
    r["gold"] = pool.label.values
    print(f"accuracy vs CLINC150 labels: {(r.label == r.gold).mean():.1%}; unparseable {(r.label == 'unparseable').mean():.1%}; "
          f"cost of these calls ${r.usd.sum():.4f}; ledger ${ledger.spent(dom.name):.4f}")
    hit = r.gold.str.replace(f"{a.domain}:", "").where(r.label == r.gold)
    wrong = r[r.label != r.gold]
    print("most common mistakes (human label -> Claude):")
    print(wrong.groupby(["gold", "label"]).size().sort_values(ascending=False).head(8).to_string())


def cmd_streams(a):
    pools, fewshot, pool, intents, prompt, prompt_base, base_fs = setup(a.domain)
    dom, base = teachers(a.domain, prompt, prompt_base, intents, pools, readonly=True)
    price = PRICES["haiku-5.5"]
    for seed in a.seeds:
        sc = D.StreamConfig(history=4000, n=8000, seed=seed, shots=1)
        stream, _ = D.build_stream(pools, sc, fewshot, launch_pool=pool)
        is_dom = stream.gold.str.startswith(f"{a.domain}:").to_numpy()
        rows = [None] * len(stream)
        for teacher, mask in ((dom, is_dom), (base, ~is_dom)):
            idx = np.flatnonzero(mask)
            recs = teacher.label_all(stream.text.iloc[idx].tolist())
            for i, r in zip(idx, recs):
                rows[i] = {k: r[k] for k in ("label", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
        tdf = pd.DataFrame(rows).rename(columns={"label": "teacher"})
        tdf.insert(0, "t", stream.t.values)
        run = Path(f"runs/clinc-{a.domain}-s{seed}")
        run.mkdir(parents=True, exist_ok=True)
        stream.to_csv(run / "stream.csv", index=False)
        tdf.to_csv(run / "teacher.csv", index=False)
        live = (stream.phase != "history").to_numpy()
        ok = (tdf.teacher.to_numpy() == stream.gold.to_numpy())
        meta = {"run": run.name, "teacher": "anthropic:claude-haiku-5-5", "seed": seed, "history": 4000, "live": 8000,
                "teacher_accuracy_live": float(ok[live].mean()), "price": "haiku-5.5", "domain": a.domain,
                "intents": list(intents)}
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{run}: launch-domain queries in live stream {int((is_dom & live).sum())}, teacher accuracy "
              f"{ok[live].mean():.1%} (domain rows {ok[is_dom & live].mean():.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label")
    l.add_argument("--domain", choices=list(DOMAINS), required=True)
    l.add_argument("--cap-usd", type=float, required=True)
    l.add_argument("--workers", type=int, default=8)
    l.add_argument("--yes", action="store_true")
    s = sub.add_parser("streams")
    s.add_argument("--domain", choices=list(DOMAINS), required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args()
    {"label": cmd_label, "streams": cmd_streams}[a.cmd](a)
