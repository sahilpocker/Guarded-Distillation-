"""Offline evaluation of a frozen bundle on a held-out stream: the gate a promotion has to pass.

Replays the live part of a seed's stream (teacher labels from the cache, no API calls) through the bundle's
gate without any online learning, and measures what the CI eval measures: overall accuracy and launch-question
accuracy against the LLM on every query, and the share of LLM calls (escalations plus expected audits).
Thresholds are the CI ones: overall >= -0.3 pts, launch >= -2 pts, calls at most 2 points above the current model's.
"""
from __future__ import annotations

import numpy as np

from ..gate import accepts, runner_up
from ..taxonomy import OOS
from .bundle import IN, Bundle
from .triage import AUDIT_P

MIN_OVERALL, MIN_LAUNCH, MAX_CALLS_OVER = -0.3, -2.0, 2.0


def eval_stream(seed: int = 1):
    from .. import data as D
    from ..teacher import LLMTeacher, build_system_prompt
    pools = D.load_pools("data/raw")
    fewshot = D.load_fewshot(pools, 1)
    stream, _ = D.build_stream(pools, D.StreamConfig(seed=seed), fewshot)
    live = stream[stream.phase != "history"].reset_index(drop=True)
    t = LLMTeacher("anthropic", "claude-haiku-5-5", build_system_prompt(pools.intents, fewshot), list(pools.intents) + [D.OOS],
                   cache_path="labels/claude-haiku-5-5.jsonl", readonly=True)
    live["teacher"] = [r["label"] for r in t.label_all(live.text.tolist())]
    return live


def evaluate(b: Bundle, live) -> dict:
    texts = live.text.tolist()
    Ps, Pi = b.scope.proba(texts), b.intent.proba(texts)
    us, ui = b.scale_scope(runner_up(Ps)), b.scale_intent(runner_up(Pi))
    if getattr(b, "router", None) is not None:             # score with the learned router, as the service does
        from .. import embed
        from .bundle import query_features
        E = embed.encode(texts)
        prior = np.full(len(b.intents), 0.15)
        X = np.array([query_features(b, texts[n], Pi[n], E[n], prior)[0] for n in range(len(texts))])
        sc = b.router.score_many(X)
        ref = np.sort(np.asarray(b.rwin)) if len(b.rwin) else np.sort(sc)
        ui = np.searchsorted(ref, sc, side="left") / len(ref)
    scope_top = np.where(Ps[:, b.scope.index[IN]] >= Ps[:, b.scope.index[OOS]], IN, OOS)
    intent_top = np.array(b.intent.universe)[Pi.argmax(1)]
    scope_ok = accepts(us, b.q0_scope)
    oos_ok = scope_ok & (scope_top == OOS)
    intent_ok = scope_ok & (scope_top == IN) & accepts(ui, b.q0_intent)
    accept = oos_ok | intent_ok
    answer = np.where(accept, np.where(oos_ok, OOS, intent_top), live.teacher.to_numpy())
    gold = live.gold.to_numpy()
    correct, llm_correct = answer == gold, live.teacher.to_numpy() == gold
    launch = live.is_launch.to_numpy()
    calls = float((~accept).mean() + AUDIT_P * accept.mean()) * 100
    return {"calls_pct": round(calls, 2), "accuracy": round(float(correct.mean() * 100), 2),
            "overall_delta": round(float((correct.mean() - llm_correct.mean()) * 100), 2),
            "launch_delta": round(float((correct[launch].mean() - llm_correct[launch].mean()) * 100), 2),
            "llm_accuracy": round(float(llm_correct.mean() * 100), 2)}


def gate(cand: dict, current: dict | None) -> list[tuple[str, str, str, bool]]:
    rows = [("Overall accuracy vs LLM only", f"{cand['overall_delta']:+.2f} pts", f">= {MIN_OVERALL:+.1f}", cand["overall_delta"] >= MIN_OVERALL),
            ("Launch-question accuracy vs LLM only", f"{cand['launch_delta']:+.2f} pts", f">= {MIN_LAUNCH:+.1f}", cand["launch_delta"] >= MIN_LAUNCH)]
    if current:
        lim = current["calls_pct"] + MAX_CALLS_OVER
        rows.append(("LLM calls", f"{cand['calls_pct']:.1f}%", f"<= {lim:.1f}% (current {current['calls_pct']:.1f}% + {MAX_CALLS_OVER:g})", cand["calls_pct"] <= lim))
    return rows
