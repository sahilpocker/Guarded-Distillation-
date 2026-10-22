"""Datasets and the replay stream.

Sources (both public, human-labelled):
  * Banking77 (PolyAI): 13,083 customer queries over 77 fine-grained banking intents.
  * CLINC150 out-of-scope queries: 1,200 queries outside any supported intent.

Before the stream starts there is a HISTORY: queries the client's existing LLM agent already
handled while we watched in shadow mode (onboarding). The teacher's labels on them train
students v1 and calibrate the starting cortisol. History costs LLM calls; it is reported as
onboarding cost, separate from the live stream.

The stream is one simulated "support inbox" in four phases:
  steady    normal traffic; the launch intents do not exist yet; 2% out-of-scope
  launch    the bank launches virtual cards and wallet pay; those intents arrive in a burst
  oos_surge a marketing campaign brings many off-topic questions (out-of-scope jumps to 20%)
  settle    traffic returns to a new normal that includes the launched product

The gold label is the dataset's human label. The system never sees it; it is used only to score.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .taxonomy import LAUNCH_INTENTS, OOS, check_complete

SOURCES = {
    "banking77_train.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    ),
    "banking77_test.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    ),
    "clinc150_data_full.json": (
        "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
        "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(raw_dir: str | Path) -> Path:
    """Fetch the raw files into raw_dir (skips files already present with the right hash)."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in SOURCES.items():
        dest = raw / name
        if dest.exists() and _sha256(dest) == sha:
            continue
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed https URLs)
            dest.write_bytes(r.read())
        got = _sha256(dest)
        if got != sha:
            raise RuntimeError(f"{name}: sha256 {got} does not match the pinned {sha}")
    return raw


@dataclass(frozen=True)
class Pools:
    banking: pd.DataFrame          # columns: text, label
    oos: pd.DataFrame              # columns: text, label (= OOS)
    intents: tuple[str, ...]       # the 77 Banking77 intents, sorted


def load_pools(raw_dir: str | Path) -> Pools:
    raw = Path(raw_dir)
    missing = [n for n in SOURCES if not (raw / n).exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {raw}; run `cortisol data` first")
    b = pd.concat([pd.read_csv(raw / "banking77_train.csv"), pd.read_csv(raw / "banking77_test.csv")])
    b = b.rename(columns={"category": "label"})[["text", "label"]]
    b["text"] = b["text"].astype(str).str.strip()
    b = b.drop_duplicates("text").reset_index(drop=True)
    j = json.loads((raw / "clinc150_data_full.json").read_text())
    oos_rows = j["oos_train"] + j["oos_val"] + j["oos_test"]
    o = pd.DataFrame({"text": [t.strip() for t, _ in oos_rows], "label": OOS}).drop_duplicates("text")
    intents = tuple(sorted(b["label"].unique()))
    check_complete(list(intents))
    return Pools(banking=b, oos=o.reset_index(drop=True), intents=intents)


@dataclass(frozen=True)
class Phase:
    name: str
    start: float           # fraction of the stream where the phase starts
    launch_share: tuple[float, float]  # share of traffic from launch intents at phase start, end
    oos_share: float


@dataclass(frozen=True)
class StreamConfig:
    history: int = 4000    # shadow-mode queries before go-live (teacher-labelled, onboarding)
    n: int = 8000          # live stream length
    shots: int = 1         # examples per label held back for the LLM prompt (never in the stream)
    seed: int = 0
    phases: tuple[Phase, ...] = field(default_factory=lambda: (
        Phase("steady", 0.00, (0.00, 0.00), 0.02),
        Phase("launch", 0.35, (0.22, 0.08), 0.02),
        Phase("oos_surge", 0.65, (0.06, 0.06), 0.20),
        Phase("settle", 0.80, (0.06, 0.06), 0.03),
    ))


def _phase_at(cfg: StreamConfig, t: int) -> tuple[Phase, float]:
    frac = t / cfg.n
    idx = max(i for i, p in enumerate(cfg.phases) if p.start <= frac)
    p = cfg.phases[idx]
    end = cfg.phases[idx + 1].start if idx + 1 < len(cfg.phases) else 1.0
    within = (frac - p.start) / max(end - p.start, 1e-9)
    return p, within


FEWSHOT_PATH = Path("data/fewshot.csv")


def choose_fewshot(pools: Pools, shots: int, seed: int = 0) -> pd.DataFrame:
    """`shots` examples per label (77 intents + out_of_scope), chosen with a fixed seed."""
    b = pools.banking.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    o = pools.oos.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    few = [b[b.label == lab].head(shots) for lab in pools.intents] + [o.head(shots)]
    return pd.concat(few).reset_index(drop=True)


def load_fewshot(pools: Pools, shots: int, path: str | Path = FEWSHOT_PATH) -> pd.DataFrame:
    """The prompt's few-shot set, chosen once and saved, so every seed shares one system prompt."""
    path = Path(path)
    if path.exists():
        fs = pd.read_csv(path, keep_default_na=False)
        if len(fs) != shots * (len(pools.intents) + 1):
            raise ValueError(f"{path} has {len(fs)} rows; it does not match --shots {shots}. Delete it to rechoose.")
        return fs
    fs = choose_fewshot(pools, shots)
    path.parent.mkdir(parents=True, exist_ok=True)
    fs.to_csv(path, index=False)
    return fs


def pool_texts(pools: Pools, fewshot: pd.DataFrame) -> pd.DataFrame:
    """Every unique Banking77 + CLINC out-of-scope text outside the few-shot set, in a fixed order
    (text, gold). Labelled once per teacher; every seed's stream is drawn from this pool."""
    df = pd.concat([pools.banking, pools.oos]).drop_duplicates("text")
    df = df[~df.text.isin(set(fewshot.text))].sample(frac=1.0, random_state=0)
    return df.rename(columns={"label": "gold"}).reset_index(drop=True)


def build_stream(pools: Pools, cfg: StreamConfig,
                 fewshot: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stream, fewshot).

    stream columns: t, text, gold, phase, is_launch. The first cfg.history rows have
    phase == "history" (drawn like the steady phase); t counts from 0 at go-live, so history
    rows have negative t.
    fewshot: cfg.shots examples per label (77 intents + out_of_scope) for the teacher prompt.
    """
    rng = np.random.default_rng(cfg.seed)
    b = pools.banking.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    o = pools.oos.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)

    if fewshot is None:
        fewshot = choose_fewshot(pools, cfg.shots)
    used = set(fewshot.text)
    b = b[~b.text.isin(used)]
    o = o[~o.text.isin(used)]

    launch_mask = b.label.isin(LAUNCH_INTENTS)
    queues = {
        "base": list(b[~launch_mask].itertuples(index=False)),
        "launch": list(b[launch_mask].itertuples(index=False)),
        "oos": list(o.itertuples(index=False)),
    }
    rows = []
    for k in range(cfg.history + cfg.n):
        t = k - cfg.history
        if t < 0:
            p, within = Phase("history", 0.0, cfg.phases[0].launch_share, cfg.phases[0].oos_share), 0.0
        else:
            p, within = _phase_at(cfg, t)
        ls = p.launch_share[0] + (p.launch_share[1] - p.launch_share[0]) * within
        u = rng.random()
        src = "oos" if u < p.oos_share else ("launch" if u < p.oos_share + ls else "base")
        if not queues[src]:
            src = "base"
        if not queues[src]:
            raise ValueError("stream longer than the data allows; lower StreamConfig.n")
        r = queues[src].pop()
        rows.append((t, r.text, r.label, p.name, src == "launch"))
    stream = pd.DataFrame(rows, columns=["t", "text", "gold", "phase", "is_launch"])
    return stream, fewshot


def phase_starts(cfg: StreamConfig) -> dict[str, int]:
    return {p.name: int(round(p.start * cfg.n)) for p in cfg.phases}
