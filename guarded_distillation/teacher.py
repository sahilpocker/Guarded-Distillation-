"""Teachers: the LLM that answers anything, labels data for the students and audits them.

Teacher labels do not depend on the replay state, so they are computed once per stream
(`gdistill label`) and every replay reads them from disk. The ledger then charges each method
only for the calls it actually made. That is why a full set of baselines costs the same as one
LLM-only pass.

Teachers:
  simulated  Deterministic stand-in: returns the gold label with probability `acc`, otherwise a
             plausible confusion (a nearby intent, or out_of_scope). Free; for building and demos.
             Results from it show the mechanism, not real LLM accuracy. Say so wherever they appear.
  anthropic  Claude via the Anthropic SDK (reads ANTHROPIC_API_KEY). System prompt is cached.
  openai     Any OpenAI-compatible endpoint: vLLM or Ollama on the A5000, OpenRouter, etc.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .taxonomy import OOS

# ----------------------------------------------------------------------------- prompt

INSTRUCTIONS = (
    "You are the triage step of a bank's customer-support agent. Classify the customer's "
    "message into exactly one intent from the list below. If the message is not about any of "
    "these banking intents, answer out_of_scope.\n"
    "Reply with the intent name only, exactly as written in the list. No other text."
)


def build_system_prompt(intents: tuple[str, ...], fewshot: pd.DataFrame | None) -> str:
    lines = [INSTRUCTIONS, "", "Intents:"]
    lines += [f"- {lab}" for lab in intents] + [f"- {OOS}"]
    if fewshot is not None and len(fewshot):
        lines += ["", "Examples:"]
        lines += [f'"{r.text}" -> {r.label}' for r in fewshot.itertuples(index=False)]
    return "\n".join(lines)


def approx_tokens(s: str) -> int:
    """Rough token count for short English text (about 4 characters per token)."""
    return max(1, round(len(s) / 4))


# Measured on the Haiku 5.5 pilot (200 calls): the 8,791-character system prompt is 3,327 tokens.
# The list of snake_case intent names tokenizes far worse than English prose.
PROMPT_CHARS_PER_TOKEN = 8791 / 3327


def approx_prompt_tokens(s: str) -> int:
    return max(1, round(len(s) / PROMPT_CHARS_PER_TOKEN))


def parse_label(raw: str, labels: list[str]) -> str:
    """Map a model reply to a label; 'unparseable' if nothing is close enough."""
    s = raw.strip().strip("`'\".").split("\n")[0].strip()
    s = s.split("->")[-1].strip()
    if s in labels:
        return s
    low = {lab.lower(): lab for lab in labels}
    if s.lower() in low:
        return low[s.lower()]
    m = difflib.get_close_matches(s.lower(), list(low), n=1, cutoff=0.85)
    return low[m[0]] if m else "unparseable"


# ----------------------------------------------------------------------------- pricing

@dataclass(frozen=True)
class Price:
    """USD per million tokens."""
    name: str
    input: float
    output: float
    cache_read: float
    cache_write: float


# $/1M tokens. Same figures as ../vaers/scripts/run_coding.py, which took them from Anthropic's
# pricing page (cache write 1.25x input, cache read 0.1x input). Re-check before quoting them.
PRICES = {
    "haiku-5.5": Price("claude-haiku-5-5", 0.10, 0.50, 0.01, 0.125),
    "sonnet-5.5": Price("claude-sonnet-5-5", 2.00, 10.00, 0.10, 2.50),
    "opus-5.5": Price("claude-opus-5-5", 4.00, 20.00, 0.40, 5.00),
    "llama-3.1-8b-or": Price("meta-llama/llama-3.1-8b-instruct via OpenRouter", 0.02, 0.04, 0.0, 0.0),
    "free": Price("local model", 0.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class CallCost:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def usd(self, p: Price) -> float:
        return (
            self.input_tokens * p.input
            + self.output_tokens * p.output
            + self.cache_read_tokens * p.cache_read
            + self.cache_write_tokens * p.cache_write
        ) / 1e6


# ----------------------------------------------------------------------------- simulated

class SimulatedTeacher:
    """Stand-in for an LLM: right on easy queries, wrong mostly on ambiguous ones.

    Real LLM mistakes concentrate on genuinely ambiguous queries (intents that overlap), not
    uniformly at random. To mimic that, a reference classifier is cross-fitted on the whole
    labelled pool (5 folds). For each query it gives p_ref(gold) and its most likely wrong label.
    The teacher errs with probability proportional to (1 - p_ref(gold)) ** sharpness, scaled so
    the average accuracy is `acc`; when it errs it gives the reference's top wrong label. Each
    text's outcome is seeded by its hash, so the same text always gets the same label.

    `sharpness=0` gives errors uniformly at random (a harsher, less realistic teacher).
    """

    name = "simulated"

    def __init__(self, labelled: pd.DataFrame, acc: float = 0.93, sharpness: float = 2.0, seed: int = 0):
        from scipy.special import softmax
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_union
        from sklearn.svm import LinearSVC

        self.acc, self.seed = acc, seed
        df = labelled.drop_duplicates("text").reset_index(drop=True)
        labels = sorted(df.label.unique())
        y = df.label.map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        P = np.zeros((len(df), len(labels)))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df.text, y):
            vec = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                             TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
            Xtr = vec.fit_transform(df.text.iloc[tr])
            clf = LinearSVC(C=0.5, random_state=seed).fit(Xtr, y[tr])
            P[np.ix_(te, clf.classes_)] = softmax(clf.decision_function(vec.transform(df.text.iloc[te])) * 5, axis=1)
        p_gold = P[np.arange(len(df)), y]
        P_wrong = P.copy()
        P_wrong[np.arange(len(df)), y] = -1
        wrong = np.array(labels)[P_wrong.argmax(1)]
        w = (1.0 - p_gold) ** sharpness if sharpness > 0 else np.ones(len(df))
        lo, hi = 0.0, 1e6  # find k with mean(min(1, k w)) = 1 - acc
        for _ in range(100):
            k = (lo + hi) / 2
            lo, hi = (k, hi) if np.minimum(1, k * w).mean() < 1 - acc else (lo, k)
        self.err_p = dict(zip(df.text, np.minimum(1, k * w)))
        self.wrong = dict(zip(df.text, wrong))

    def _u(self, text: str) -> float:
        h = int(hashlib.sha1(f"{self.seed}|{text}".encode(), usedforsecurity=False).hexdigest()[:12], 16)
        return np.random.default_rng(h).random()

    def label(self, text: str, gold: str) -> str:
        return self.wrong[text] if self._u(text) < self.err_p[text] else gold


# ----------------------------------------------------------------------------- LLM teachers

class CapExceeded(RuntimeError):
    """The spend cap would be exceeded by the next call."""


class FatalAPIError(RuntimeError):
    """A non-retryable API error (bad key, bad request, unknown model)."""


class Ledger:
    """Cumulative USD spent per teacher, persisted to a JSON file shared across runs.

    Spend is computed from the usage the API returns. `reserve` holds back a worst-case cost
    for each in-flight call, so concurrent workers cannot overshoot the cap.
    """

    def __init__(self, path: str | Path, cap_usd: float | None = None):
        self.path, self.cap = Path(path), cap_usd
        self._lock = threading.Lock()
        self._reserved = 0.0
        self.by_model: dict[str, float] = {}
        if self.path.exists():
            self.by_model = json.loads(self.path.read_text()).get("by_model", {})

    def spent(self, model: str) -> float:
        return self.by_model.get(model, 0.0)

    def reserve(self, model: str, worst: float) -> bool:
        with self._lock:
            if self.cap is not None and self.spent(model) + self._reserved + worst > self.cap:
                return False
            self._reserved += worst
            return True

    def settle(self, model: str, worst: float, actual: float) -> None:
        with self._lock:
            self._reserved -= worst
            self.by_model[model] = self.spent(model) + actual
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"by_model": self.by_model}, indent=2))


class LLMTeacher:
    """Labels texts with a real model, with an append-only on-disk cache (safe to interrupt).

    Failed calls are never cached. A cache hit costs nothing. With a ledger, every call is
    charged from the usage the API returned and the run stops before the cap is exceeded.
    With readonly=True nothing is ever sent: missing texts raise instead.
    """

    def __init__(self, provider: str, model: str, system_prompt: str, labels: list[str],
                 cache_path: Path, base_url: str | None = None, max_workers: int = 8,
                 price: Price | None = None, ledger: Ledger | None = None, readonly: bool = False,
                 client_kwargs: dict | None = None, tries: int = 4):
        self.provider, self.model = provider, model
        self.system_prompt, self.labels = system_prompt, labels
        self.cache_path = Path(cache_path)
        self.base_url = base_url
        self.max_workers = max_workers
        self.price, self.ledger, self.readonly = price, ledger, readonly
        self.client_kwargs, self.tries = client_kwargs or {}, tries
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.cap_hit, self.fatal, self.last_error = False, None, None
        self._client = None
        self.name = f"{provider}:{model}"

    def _key(self, text: str) -> str:
        sp = hashlib.sha1(self.system_prompt.encode(), usedforsecurity=False).hexdigest()[:10]
        return hashlib.sha1(f"{self.name}|{sp}|{text}".encode(), usedforsecurity=False).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line from an interrupted run
                out[rec["key"]] = rec
        return out

    def _client_get(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(**self.client_kwargs)
            elif self.provider == "openai":
                from openai import OpenAI
                self._client = OpenAI(base_url=self.base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"))
            else:
                raise ValueError(f"unknown provider {self.provider}")
        return self._client

    def _call(self, text: str) -> dict:
        c = self._client_get()
        if self.provider == "anthropic":
            # No temperature (sampling parameters are rejected by current models) and thinking
            # off: the reply is one intent name and must fit in max_tokens.
            # Sonnet 5.5 rejects "disabled": "between_tools" is its way of not thinking before the reply.
            thinking = {"type": "between_tools"} if "sonnet" in self.model else {"type": "disabled"}
            r = c.messages.create(
                model=self.model, max_tokens=24, thinking=thinking,
                system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": text}],
            )
            raw = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            u = r.usage
            usage = dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                         cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                         cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0)
        else:
            r = c.chat.completions.create(
                model=self.model, max_tokens=24, temperature=0,
                messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            )
            raw = r.choices[0].message.content or ""
            u = r.usage
            usage = dict(input_tokens=getattr(u, "prompt_tokens", 0), output_tokens=getattr(u, "completion_tokens", 0),
                         cache_read_tokens=0, cache_write_tokens=0)
        return {"raw": raw, "label": parse_label(raw, self.labels), **usage}

    def _worst(self, text: str) -> float:
        """Upper bound on one call's cost, held back in the ledger while the call is in flight."""
        p = self.price
        tokens = 1.5 * (approx_prompt_tokens(self.system_prompt) + approx_tokens(text) + 50)
        return (tokens * max(p.input, p.cache_write) + 24 * p.output) / 1e6

    def _call_retry(self, text: str, tries: int | None = None) -> dict | None:
        """One record, or None (cap hit, fatal error, or transient failure: nothing is cached)."""
        import time
        tries = tries or self.tries
        if self._stop.is_set():
            return None
        worst = self._worst(text) if self.price else 0.0
        if self.ledger and not self.ledger.reserve(self.name, worst):
            self.cap_hit = True
            self._stop.set()
            return None
        actual = 0.0
        try:
            for i in range(tries):
                try:
                    rec = self._call(text)
                    if self.price:
                        actual = CallCost(rec["input_tokens"], rec["output_tokens"], rec["cache_read_tokens"],
                                          rec["cache_write_tokens"]).usd(self.price)
                    rec["usd"] = actual
                    return rec
                except Exception as e:
                    self.last_error = f"{type(e).__name__}: {e}"
                    if isinstance(e, ImportError) or getattr(e, "status_code", None) in (400, 401, 403, 404):
                        self.fatal = f"{type(e).__name__}: {e}"
                        self._stop.set()
                        return None
                    if i == tries - 1:
                        return None
                    time.sleep(2 ** i)
        finally:
            if self.ledger:
                self.ledger.settle(self.name, worst, actual)
        return None

    def label_all(self, texts: list[str], progress: bool = True) -> list[dict]:
        cache = self._load_cache()
        todo = [t for t in dict.fromkeys(texts) if self._key(t) not in cache]
        if todo and self.readonly:
            raise RuntimeError(f"{len(todo):,} of {len(set(texts)):,} texts are not in {self.cache_path}; "
                               "run `gdistill label-pool` for this teacher first")
        if todo:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            done = 0
            with self.cache_path.open("a") as f:
                def store(t, rec):
                    nonlocal done
                    if rec is None:
                        return
                    rec = {"key": self._key(t), "text": t, **rec}
                    with self._lock:
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        cache[rec["key"]] = rec
                    done += 1
                    if progress and done % 250 == 0:
                        print(f"  labelled {done}/{len(todo)}", flush=True)

                if self.provider == "anthropic" and not self.cache_path.stat().st_size:
                    store(todo[0], self._call_retry(todo[0]))   # first call alone: writes the prompt cache
                    todo = todo[1:]
                with ThreadPoolExecutor(self.max_workers) as ex:
                    futs = {ex.submit(self._call_retry, t): t for t in todo}
                    for fut in as_completed(futs):
                        store(futs[fut], fut.result())
        if self.fatal:
            raise FatalAPIError(self.fatal)
        if self.cap_hit:
            raise CapExceeded(f"spend cap reached at ${self.ledger.spent(self.name):.4f}; "
                              f"{sum(self._key(t) not in cache for t in set(texts)):,} texts unlabelled")
        missing = [t for t in texts if self._key(t) not in cache]
        if missing:
            raise RuntimeError(f"{len(missing):,} calls failed after retries (not cached); last error: {self.last_error}")
        return [cache[self._key(t)] for t in texts]


# ----------------------------------------------------------------------------- precompute

def precompute(stream: pd.DataFrame, teacher, system_prompt: str) -> pd.DataFrame:
    """Teacher label and token usage for every stream row (t, teacher, input_tokens, ...)."""
    if isinstance(teacher, SimulatedTeacher):
        sys_tokens = approx_prompt_tokens(system_prompt)
        recs = []
        for i, r in enumerate(stream.itertuples(index=False)):
            # Priced as a cached system prompt: the first call writes the cache, later calls read it.
            recs.append(dict(teacher=teacher.label(r.text, r.gold),
                             input_tokens=approx_tokens(r.text) + 10,   # pilot: 23.8 uncached tokens per call
                             output_tokens=10,                          # pilot: 10.4 (intent names are long)
                             cache_read_tokens=sys_tokens if i else 0,
                             cache_write_tokens=0 if i else sys_tokens))
        out = pd.DataFrame(recs)
    else:
        res = teacher.label_all(stream.text.tolist())
        out = pd.DataFrame([{k: r[k] for k in ("label", "input_tokens", "output_tokens",
                                                "cache_read_tokens", "cache_write_tokens")} for r in res])
        out = out.rename(columns={"label": "teacher"})
    out.insert(0, "t", stream.t.values)
    return out
