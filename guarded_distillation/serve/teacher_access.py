"""Teacher access for the service: the label cache first, a live Claude call only on a miss, under a daily cap."""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

import pandas as pd

from ..data import FEWSHOT_PATH
from ..teacher import PRICES, CapExceeded, FatalAPIError, Ledger, LLMTeacher, build_system_prompt
from ..taxonomy import OOS

MODEL = "claude-haiku-5-5"


class DailyLedger(Ledger):
    """Spend cap that resets every UTC day: the ledger key carries the date."""

    def _k(self, model: str) -> str:
        return f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}|{model}"

    def spent(self, model: str) -> float:
        return super().spent(self._k(model))

    def reserve(self, model: str, worst: float) -> bool:
        return super().reserve(self._k(model), worst)

    def settle(self, model: str, worst: float, actual: float) -> None:
        super().settle(self._k(model), worst, actual)

    def total(self) -> float:
        return float(sum(self.by_model.values()))


class DbLedger:
    """Daily spend cap that lives in the store, so a restart does not reset it. Same interface as Ledger."""

    def __init__(self, store, cap_usd: float):
        self.store, self.cap = store, cap_usd
        self._lock = threading.Lock()
        self._reserved = 0.0

    @staticmethod
    def _day() -> str:
        return f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}"

    def spent(self, model: str) -> float:
        return self.store.spend_today(self._day())

    def reserve(self, model: str, worst: float) -> bool:
        with self._lock:
            if self.cap is not None and self.store.spend_today(self._day()) + self._reserved + worst > self.cap:
                return False
            self._reserved += worst
            return True

    def settle(self, model: str, worst: float, actual: float) -> None:
        with self._lock:
            self._reserved -= worst
        if actual:
            self.store.spend_add(self._day(), model, actual)

    def total(self) -> float:
        return self.store.spend_total()


class TeacherUnavailable(RuntimeError):
    pass


class TeacherAccess:
    def __init__(self, intents: list[str], cache: str | Path = "labels/claude-haiku-5-5.jsonl", cap_usd: float = 1.0,
                 ledger: "str | Path | DbLedger" = "data/service_spend.json", fewshot: str | Path = FEWSHOT_PATH, live: bool = True):
        self.cache_path = Path(cache)
        self.by_text: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.by_text[r["text"]] = r
        self.cap = cap_usd
        self.ledger = ledger if isinstance(ledger, DbLedger) else DailyLedger(ledger, cap_usd)
        self.live = live
        self._lock = threading.Lock()
        fs = pd.read_csv(fewshot, keep_default_na=False) if Path(fewshot).exists() else None
        self._prompt = build_system_prompt(tuple(intents), fs)
        self._llm = None
        self.cache_hits = self.api_calls = 0
        self.intents = intents

    def _teacher(self) -> LLMTeacher:
        if self._llm is None:
            self._llm = LLMTeacher("anthropic", MODEL, self._prompt, list(self.intents) + [OOS], cache_path=self.cache_path,
                                   max_workers=1, price=PRICES["haiku-5.5"], ledger=self.ledger,
                                   client_kwargs={"timeout": 12.0, "max_retries": 0}, tries=1)       # fail fast: a request is waiting
        return self._llm

    def spent_today(self) -> float:
        return self.ledger.spent(f"anthropic:{MODEL}")

    def label(self, text: str, allow_live=None) -> tuple[str, str, float]:
        """(label, source, usd): source is 'cache' or 'api'. Raises TeacherUnavailable.
        `allow_live()` is asked before a live call (per-visitor and global rate limits); False means no call."""
        r = self.by_text.get(text)
        if r and r.get("label") != "unparseable":
            self.cache_hits += 1
            return r["label"], "cache", 0.0
        if not self.live:
            raise TeacherUnavailable("not in the label cache and live calls are off")
        if allow_live is not None and not allow_live():
            raise TeacherUnavailable("rate limit on live Claude calls; try again later")
        with self._lock:
            try:
                rec = self._teacher().label_all([text], progress=False)[0]
            except CapExceeded as e:
                raise TeacherUnavailable(f"daily spend cap (${self.cap:.2f}) reached") from e
            except (FatalAPIError, RuntimeError) as e:
                raise TeacherUnavailable(str(e)[:160]) from e
        self.by_text[text] = rec
        self.api_calls += 1
        return rec["label"], "api", float(rec.get("usd", 0.0))
