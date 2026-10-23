# cortisol-mvp

An agent pipeline that compiles its own repeated LLM decisions into small CPU models, and keeps
them honest without anyone watching.

The LLM answers anything new. Steps that keep repeating get distilled into cheap models trained on
the LLM's own answers. A **cortisol field** (a global level plus a local level for each region of
query space) decides, query by query, whether a cheap model may answer or the LLM must; where to
spend audits; and when to retrain. Misses and surprises raise it; calm periods let it fall.

Demo: a bank's support triage (scope, 77 intents, routing team, urgency), replayed on 8,000 real
customer queries from Banking77 and CLINC150, through a product launch and an off-topic surge.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Homebrew Python 3.11, same as ../vaers
pip install -e ".[dev]"
python -m pytest -q
cortisol demo                # downloads data, simulated teacher, 5 methods, report (~3 min, free)
open runs/simulated-s0/replay.html
```

## Results (simulated teacher, 3 seeds)

| Method | Accuracy vs LLM only | LLM calls | Product-launch questions |
|---|---|---|---|
| LLM only | baseline | 100% | 94.5% |
| Static gate | −1.9 pts | 46% | 76% |
| Static gate + retraining | −0.25 pts | 47% | 93% |
| Global cortisol | −0.26 pts | 48% | 93% |
| Cortisol field | −0.35 pts | 47% | 92% |

About half the LLM calls at a third of a point of accuracy, with no human step through a launch
and a surge. Retraining does most of the work; the local field has not yet shown an advantage on
this stream. See docs/RESULTS.md, and docs/ROADMAP.md for the experiments that test it.

## Recommended setting (Claude Haiku 5.5 as the teacher)

Chosen on seeds 0-2 from an 18-setting grid: the fewest LLM calls such that overall accuracy is
within 0.3 pts of LLM only **and** launch-question accuracy within 2 pts, on every seed
(docs/RESULTS.md). **Static gate + retraining, MiniLM student, `alpha_intent` 0.03.**

| Seeds 3-5 (held out) | Accuracy | LLM calls | Cost / 1,000 queries | Cheap-path errors / 1k | Launch questions |
|---|---|---|---|---|---|
| LLM only (Haiku) | 80.8% | 100% | $0.041 | 0 | 84.8% |
| Recommended, no balancing | 81.3% (+0.45 pts) | 51% (46-58) | $0.021 | 43 | 81.4% (-3.4 pts) |
| Recommended + balanced retraining | +0.57 pts | 50% (46-58) | n/a | 42 | -0.3 pts |

**Held-out caveat:** with the recommended setting as first chosen (no class balancing), overall
accuracy and call savings held up but launch-question accuracy did not (-3.4 pts, two of three seeds
missed the -2 pt bar). Class-balanced retraining (a new intent is weighted up to the size of its
nearest old neighbour until it has 50 labels; `--preset recommended` now turns it on) fixes most of
it: on seeds 3-5, 50% LLM calls, +0.57 pts overall, launch -0.3 pts, 42 cheap-path errors / 1k. That
fix was found after seeing the seeds 3-5 result, so those numbers are no longer a clean held-out test
of the preset.

![Operating points](docs/operating-points.svg)

```bash
pip install -e ".[llm,sbert]"
cortisol replay --run runs/<run> --preset recommended
cortisol report --run runs/<run> --focus retrain
```

**The cortisol field (the local, per-region part) is experimental.** It is on the trade-off
curve at one point only and no experiment has shown it beating the static gate with retraining.

**Semantic cortisol (experimental, `semantic` method):** per-intent stress on a graph of intents,
instead of k-means cells. On seeds 3-5, at 57% LLM calls it reaches launch accuracy +0.4 pts vs
LLM only and 0.0 in the first 600 launch-phase queries; the balanced plain gate at the same calls is
-0.3 and -2.4. The gain comes from per-intent error residuals spread to graph neighbours; the new-intent,
fit-check and LLM-sensor parts add little (the LLM sensor as specified never fired). Three seeds, tuned
on 0-2 with more freedom than the baseline; see docs/RESULTS.md. **On a CLINC150 domain holdout (a whole
new 15-intent domain as the launch event) it does not carry over cleanly:** better than the matched
balanced gate on the adjacent `credit_cards` domain (launch -2.1 vs -3.2 pts), no better on `travel`
(-3.2 vs -3.8), and no setting meets the -2 pt launch bar there.

## Real teachers

```bash
pip install -e ".[llm]"
# local model behind any OpenAI-compatible server (vLLM, Ollama), free
cortisol label --teacher openai --model <name> --base-url http://host:8000/v1 --price free --run runs/local-s0 --yes
cortisol replay --run runs/local-s0
cortisol report --run runs/local-s0
```

Claude works too (`--teacher anthropic --model claude-haiku-5-5`, key in `.env`); without `--yes` it
only prints the cost estimate. To label a whole dataset once with a spend cap, then replay any seed
for free: `cortisol label-pool ... --cap-usd 5 --yes`, then `cortisol label --labels labels/<model>.jsonl`.

Teacher labels do not depend on the method, so each stream is labelled once and every method
replays from the cache. Each method is charged only for the calls it makes.

## Repository

| path | what |
|---|---|
| `cortisol/engine.py` | the replay loop: gates, escalation, audits, cortisol, retraining |
| `cortisol/gate.py` | doubt, leak bound, the cortisol field, `aci_update` |
| `cortisol/student.py`, `space.py`, `embed.py` | compiled models (TF-IDF or MiniLM); the map space and its cells |
| `cortisol/teacher.py` | simulated, Anthropic and OpenAI-compatible teachers |
| `cortisol/data.py`, `taxonomy.py` | datasets, the drift stream, routing tables |
| `cortisol/report.py`, `templates/replay.html` | REPORT.md, results.json, the replay dashboard |
| `docs/THEORY.md` | lessons, the leak guarantee and its proof sketch, the research question |
| `docs/DESIGN.md` | why each piece is built the way it is |
| `docs/RESULTS.md`, `docs/ROADMAP.md` | numbers so far; what to run next |
| `scripts/` | `grid.py` (run the operating-point grid), `operating_points.py` (select, plot), `seed_table.py` |

## Data

Banking77 (PolyAI, CC-BY-4.0) and CLINC150 out-of-scope queries (CC-BY-3.0), downloaded from their
GitHub repositories with pinned SHA-256 hashes.
