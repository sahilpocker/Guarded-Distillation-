"""LLM sensor for semantic cortisol: which known intents could an unfamiliar cluster be confused with?

One short Claude call per cluster (a few hundred tokens). Answers are cached by the cluster's
examples, so reruns are free, and spend is capped by a ledger of its own.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .teacher import CallCost, CapExceeded, Ledger, PRICES

MODEL = "claude-haiku-5-5"
SYSTEM = (
    "You monitor an intent classifier for a bank's customer-support inbox. You are given a group of "
    "customer messages that the classifier found unfamiliar, and the list of intents it knows. Name "
    "the known intents a classifier would most likely confuse these messages with (at most 5), most "
    'likely first. Reply with a JSON list of intent names only, e.g. ["card_arrival", "lost_or_stolen_card"].'
)


class HaikuSensor:
    def __init__(self, cache: str | Path = "labels/sensor-haiku.jsonl", ledger: str | Path = "labels/sensor-spend.json",
                 cap_usd: float = 0.50):
        self.cache_path, self.ledger = Path(cache), Ledger(ledger, cap_usd)
        self.cache: dict[str, list[str]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.cache[r["key"]] = r["intents"]
        self.calls = 0
        self._client = None

    def spent(self) -> float:
        return self.ledger.spent(MODEL)

    def ask(self, examples: list[str], intents: list[str]) -> list[str]:
        key = hashlib.sha256(json.dumps([examples, intents]).encode()).hexdigest()[:20]
        if key in self.cache:
            return self.cache[key]
        user = ("Known intents: " + ", ".join(intents) + "\n\nUnfamiliar messages:\n"
                + "\n".join(f"- {e}" for e in examples))
        p = PRICES["haiku-5.5"]
        worst = (1.5 * (len(SYSTEM) + len(user)) / 3.5 * max(p.input, p.cache_write) + 200 * p.output) / 1e6
        if not self.ledger.reserve(MODEL, worst):
            raise CapExceeded("sensor cap reached")
        try:
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic()
            r = self._client.messages.create(model=MODEL, max_tokens=200, thinking={"type": "disabled"},
                                             system=SYSTEM, messages=[{"role": "user", "content": user}])
        except Exception:
            self.ledger.settle(MODEL, worst, 0.0)
            raise
        u = r.usage
        usd = CallCost(u.input_tokens, u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0,
                       getattr(u, "cache_creation_input_tokens", 0) or 0).usd(p)
        self.ledger.settle(MODEL, worst, usd)
        raw = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        names = [n for n in intents if n in raw]
        names.sort(key=raw.index)
        names = names[:5]
        self.cache[key] = names
        self.calls += 1
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "intents": names, "raw": raw, "usd": usd}) + "\n")
        return names
