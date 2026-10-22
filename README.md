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

Chosen on seeds 0-2 from an 18-setting grid and checked on seeds 3-5 (docs/RESULTS.md):
**global cortisol + retraining, MiniLM student, `alpha_intent` 0.08**.

| | Accuracy | LLM calls | Cost / 1,000 queries | Cheap-path errors / 1k |
|---|---|---|---|---|
| LLM only (Haiku) | 80.8% | 100% | $0.041 | 0 |
| Recommended, seeds 3-5 | 81.2% (+0.34 pts) | 60% (57-64) | $0.024 | 55 |

![Operating points](docs/operating-points.svg)

```bash
pip install -e ".[llm,sbert]"
cortisol replay --run runs/<run> --preset recommended
cortisol report --run runs/<run> --focus global
```

**The cortisol field (the local, per-region part) is experimental.** It was not on the
trade-off curve, and no experiment has yet shown it beating global cortisol (E3, E4 in
docs/ROADMAP.md test where it should). Launch-question accuracy is about 2 points below LLM only
under this setting.

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
