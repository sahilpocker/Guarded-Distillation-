"""Synthetic test data: fresh wording for known intents, in controlled styles, from NVIDIA-hosted Nemotron (reasoning switched off: it would otherwise spend the token budget thinking).

    python scripts/synth.py generate --out data/synth/pilot.jsonl        # NVIDIA API (NVIDIA_API_KEY in .env)
    python scripts/synth.py curate   --inp data/synth/pilot.jsonl        # MiniLM dedup + Haiku agreement check (Claude calls, capped)
    python scripts/synth.py probe    --bundle models/candidates/v2       # how a bundle's gate behaves on it, by style

A pilot: 15 intents x 4 styles x 5 messages = 300. The generator is told the intent and shown five real examples; every
item is then labelled blind by Haiku and kept only when Haiku agrees with the generator's intent. Synthetic text is cleaner
than real traffic, so results on it are a robustness suite, never mixed into the human-labelled numbers.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from guarded_distillation.cli import _load_env  # noqa: E402

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"        # many catalogue entries 404 for an account; this one is served (checked 23 Oct 2026)
BASE_URL = "https://integrate.api.nvidia.com/v1"
INTENTS = ["getting_virtual_card", "get_disposable_virtual_card", "virtual_card_not_working", "disposable_card_limits", "apple_pay_or_google_pay",
           "card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "compromised_card", "card_not_working", "pending_card_payment",
           "declined_card_payment", "exchange_rate", "top_up_failed", "transfer_not_received_by_recipient"]
STYLES = {
    "plain": "clear and neutral, one sentence, ordinary grammar",
    "casual": "casual texting style: lowercase, slang, no end punctuation, and one or two natural typos",
    "context": "two or three sentences: a little background about what happened, then the question",
    "terse": "very short, under seven words, like a search query or a rushed message",
}
PER_CALL = 5
SYSTEM = ("You write realistic customer messages for a bank's customer-support chat, to test a classifier. "
          "Reply with a JSON list of strings and nothing else.")


def prompt(intent, examples, style):
    ex = "\n".join(f"- {e}" for e in examples)
    return (f"Intent: {intent.replace('_', ' ')}\nReal customer messages with this intent:\n{ex}\n\n"
            f"Write {PER_CALL} NEW customer messages with exactly this intent and no other (do not mix in a different topic).\n"
            f"Style: {STYLES[style]}.\nDo not reuse the wording of the examples. Return a JSON list of {PER_CALL} strings.")


def parse_list(raw: str) -> list[str]:
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        v = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        v = re.findall(r'"([^"\n]{3,300})"', raw)
    return [x.strip() for x in v if isinstance(x, str) and x.strip()]


def cmd_generate(a):
    import time
    from openai import OpenAI, RateLimitError
    _load_env()
    from guarded_distillation import data as D
    pools = D.load_pools("data/raw")
    rng = np.random.default_rng(0)
    client = OpenAI(base_url=BASE_URL, api_key=os.environ["NVIDIA_API_KEY"].strip(), timeout=90, max_retries=0)
    jobs = []
    for intent in INTENTS:
        pool = pools.banking[pools.banking.label == intent].text.tolist()
        for style in STYLES:
            jobs.append((intent, style, [pool[i] for i in rng.choice(len(pool), 5, replace=False)]))

    def run(job):
        intent, style, ex = job
        for wait in (2, 5, 10, 20, 40, 60):            # the free endpoint rate-limits: back off and retry
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0.8, max_tokens=500,
                                                   extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                                                   messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(intent, ex, style)}])
                break
            except RateLimitError:
                time.sleep(wait)
        else:
            raise RuntimeError(f"rate limited too long on {intent}/{style}")
        items = parse_list(r.choices[0].message.content or "")[:PER_CALL]
        return [{"text": t, "intent": intent, "style": style, "generator": MODEL} for t in items], r.usage.total_tokens

    rows, toks = [], 0
    with ThreadPoolExecutor(2) as ex:
        for items, t in ex.map(run, jobs):
            rows += items
            toks += t
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} generated from {len(jobs)} calls, {toks:,} tokens -> {a.out}")


def cmd_curate(a):
    from guarded_distillation import data as D, embed
    from guarded_distillation.teacher import PRICES, CapExceeded, Ledger, LLMTeacher, build_system_prompt
    _load_env()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    pools = D.load_pools("data/raw")
    real = set(pools.banking.text) | set(pools.oos.text)
    df["exact_dup"] = df.text.str.lower().duplicated(keep="first")
    df["in_real"] = df.text.isin(real)
    E = embed.encode(df.text.tolist())
    sim = E @ E.T
    np.fill_diagonal(sim, 0)
    near = np.zeros(len(df), bool)
    for i in range(len(df)):                       # a later near-duplicate of an earlier item of the same intent
        same = np.flatnonzero((df.intent.to_numpy()[:i] == df.intent.iloc[i]) & (sim[i, :i] >= a.dedup))
        near[i] = len(same) > 0
    df["near_dup"] = near
    fewshot = D.load_fewshot(pools, 1)
    teacher = LLMTeacher("anthropic", "claude-haiku-5-5", build_system_prompt(pools.intents, fewshot), list(pools.intents) + [D.OOS],
                         cache_path="labels/claude-haiku-5-5-synth.jsonl", max_workers=6, price=PRICES["haiku-5.5"],
                         ledger=Ledger("labels/spend.json", a.cap_usd))
    try:
        recs = teacher.label_all(df.text.tolist(), progress=False)
    except CapExceeded as e:
        sys.exit(f"STOPPED: {e}")
    df["haiku"] = [r["label"] for r in recs]
    df["agree"] = df.haiku == df.intent
    df["keep"] = df.agree & ~df.exact_dup & ~df.near_dup & ~df.in_real
    df["usd"] = [r.get("usd", 0.0) for r in recs]
    out = Path(a.inp).with_name(Path(a.inp).stem + "-curated.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in df.drop(columns=["usd"]).to_dict("records")) + "\n", encoding="utf-8")
    print(f"{len(df)} items: exact dups {int(df.exact_dup.sum())}, already in the real data {int(df.in_real.sum())}, near-dups {int(df.near_dup.sum())}, "
          f"Haiku disagrees {int((~df.agree).sum())}; kept {int(df.keep.sum())}. Haiku cost ${df.usd.sum():.4f} -> {out}")
    print("\nHaiku agreement by style:\n" + df.groupby("style").agree.mean().round(3).to_string())
    print("\nHaiku agreement by intent:\n" + df.groupby("intent").agree.mean().round(2).sort_values().to_string())


def cmd_probe(a):
    from guarded_distillation import embed
    from guarded_distillation.gate import accepts, runner_up
    from guarded_distillation.serve.bundle import IN, Bundle
    df = pd.DataFrame([json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines()])
    df = df[df.keep]
    b = Bundle.load(a.bundle)
    texts = df.text.tolist()
    Ps, Pi = b.scope.proba(texts), b.intent.proba(texts)
    us, ui = b.scale_scope(runner_up(Ps)), b.scale_intent(runner_up(Pi))
    top = np.array(b.intent.universe)[Pi.argmax(1)]
    scope_in = Ps[:, b.scope.index[IN]] >= Ps[:, b.scope.index["out_of_scope"]]
    accept = accepts(us, b.q0_scope) & scope_in & accepts(ui, b.q0_intent)
    df["student_ok"] = top == df.intent.to_numpy()
    df["accepted"] = accept
    df["leak"] = df.accepted & ~df.student_ok
    g = df.groupby("style").agg(n=("text", "size"), student_acc=("student_ok", "mean"), accepted=("accepted", "mean"), leak=("leak", "mean"))
    print(f"bundle {b.version}\n" + g.round(3).to_string())
    k = df.groupby("intent").agg(n=("text", "size"), student_acc=("student_ok", "mean"), accepted=("accepted", "mean")).round(2)
    print("\nby intent:\n" + k.sort_values("student_acc").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", default="data/synth/pilot.jsonl")
    c = sub.add_parser("curate")
    c.add_argument("--inp", default="data/synth/pilot.jsonl")
    c.add_argument("--dedup", type=float, default=0.92)
    c.add_argument("--cap-usd", type=float, default=1.5)
    p = sub.add_parser("probe")
    p.add_argument("--inp", default="data/synth/pilot-curated.jsonl")
    p.add_argument("--bundle", default="models/candidates/v2")
    a = ap.parse_args()
    {"generate": cmd_generate, "curate": cmd_curate, "probe": cmd_probe}[a.cmd](a)
