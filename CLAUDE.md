# guarded-distillation

Sahil's product MVP: an agent pipeline that compiles its own repeated LLM decisions into cheap
CPU models and keeps them honest without anyone watching. A **cortisol field** (global + local
uncertainty level) decides, query by query, whether a compiled model may answer or the LLM must,
where to audit, and when to retrain. Demo domain: banking-support triage on Banking77 + CLINC150.

Sister project: `../vaers` (same idea on VAERS pharmacovigilance; its week-1 lesson was
"retraining is what restores accuracy under drift; gating alone does not". This repo confirms it.)

## Status (22 Oct 2026)

- Engine, CLI, tests and the replay dashboard work end to end with a **simulated** teacher.
- `guarded_distillation.gate.aci_update` is implemented (one line, spec in tests/test_aci.py, lesson 2 in
  docs/THEORY.md). All methods run.
- E1 done with Claude Haiku 5.5 (22 Oct 2026, $0.58 total, cap $5): the whole pool is labelled in
  `labels/claude-haiku-5-5.jsonl`; seeds 0-2 read it for free (`gdistill label --labels ...`).
  Haiku is only ~80.5% accurate vs human labels on this task; compiled methods match it and cut
  LLM calls ~27-31%. alpha_intent 0.05 chosen on seed 0 (field: 63% calls). E2 (MiniLM student) is done and
  mixed: more answered but more leaks; the field shows no advantage. Results: docs/RESULTS.md.
- **Recommended setting** (picked on seeds 0-2: fewest LLM calls with overall accuracy >= -0.3 pts and
  launch accuracy >= -2 pts on every seed): static gate + retraining, MiniLM student, alpha_intent 0.03
  (`gdistill replay --preset recommended`, report with `--focus retrain`; constants in
  `engine.RECOMMENDED`). Seeds 3-5: ~51% LLM calls, +0.45 pts overall, ~43 cheap-path errors / 1k, but
  **launch accuracy -3.4 pts vs Haiku (two of three seeds miss the -2 bar)**: the pick did not hold on
  the drift criterion and has not been re-picked on held-out seeds. **The cortisol field is
  experimental**: on the trade-off curve at one point only, never beats the static gate at the same
  student and alpha. Do not present it as the product default or claim a local advantage until E3/E4
  show one. Next: E3/E4 in docs/ROADMAP.md, Llama via OpenRouter, TF-IDF settings on seeds 3-5.
- **Semantic cortisol** (22-23 Oct 2026; `guarded_distillation/semantic.py`, method `semantic`, `scripts/semantic.py`,
  results in docs/RESULTS.md): per-intent stress on an intent graph. Class-balanced retraining
  (`EngineConfig.balance`, now in the recommended preset) lifts the plain gate's launch Δ from -3.4 to -0.3
  (seeds 3-5). Semantic cortisol beats that balanced gate on launch accuracy at matched calls (57% calls:
  +0.44 vs -0.26 launch, 0.00 vs -2.41 first-600). The driver is per-intent residual + neighbour spread;
  the LLM sensor never fired as specified (relaxed trigger: no gain; spent $0.0017 of its $0.50 cap in
  `labels/sensor-spend.json`). CLINC150 domain holdout done (scripts/clinc_domain.py, labels/claude-haiku-5-5-{credit_cards,travel}.jsonl, ~$0.21): semantic
  cortisol beats the matched balanced gate on adjacent `credit_cards` (launch -2.1 vs -3.2), not on distant `travel`; nothing meets
  the -2 pt launch bar on a whole new domain. Ledger total $0.79.
  Follow-up `sem_qfit` 0.5 (per-query fit check on the gate level): small consistent gain, travel still ~-3 launch.
- **Routing benchmark** (docs/RESULTS.md, `scripts/routing.py`, `guarded_distillation/router.py`): learned router >
  margin at matched calls on all three streams (Banking77 launch, CLINC credit_cards, travel); max probability is a better
  static score than margin; version disagreement adds nothing; OCL (third_party/, `scripts/ocl_adapter.py`) has a cliff.
  Not yet the default (`EngineConfig.router` is "margin"; "learned" needs the [sbert] extra).
- **Ops layer** (in progress): CI is live (.github/workflows/ci.yml, eval gate vs docs/eval_baseline.json); the triage service
  (`gdistill serve`, guarded_distillation/serve/) and Dockerfile work locally (296 MiB at 0.1 CPU); not deployed.
  HF Spaces Docker needs a paid plan; Render free (512 MB) is the candidate host.
- **No Claude API calls without Sahil's explicit go-ahead for that specific run.** His key lives in
  `.env` (gitignored, loaded by the CLI); do not read it, print it, or touch `../vaers/.env`.
  Paid calls need `--cap-usd`; spend is in `labels/spend.json`.

### Results so far (simulated teacher, 3 seeds, 8,000 live queries each; Haiku results are in docs/RESULTS.md)

| Method | Accuracy vs LLM only | LLM calls | Launch questions | Cheap-path errors / 1k |
|---|---|---|---|---|
| LLM only | baseline (92.5–92.9%) | 100% | 94.5% | 0 |
| Static gate | −1.9 pts | 46% | 76% | 34 |
| Static gate + retraining | −0.25 pts | 47% | 93% | 13 |
| Global cortisol | −0.26 pts | 48% | 93% | 13 |
| Cortisol field (full) | −0.35 pts | 47% | 92% | 16 |

Reading: compiling + retraining halves LLM calls at about −0.3 pts. **Retraining does the heavy
lifting.** The local field has not yet earned its complexity on this stream (new intents get
learned within ~60 labels, so there is little time for local stress to matter). Do not claim a
local-cortisol advantage until an experiment shows one (E3, E4 in the roadmap). Details:
docs/RESULTS.md.

## Layout

- `guarded_distillation/data.py` – downloads (pinned SHA-256), stream builder: shadow-mode history, then
  phases steady → launch (virtual cards + wallet pay intents appear) → oos_surge → settle
- `guarded_distillation/taxonomy.py` – labels, launch intents, T0 routing and urgency tables
- `guarded_distillation/teacher.py` – simulated teacher (errors concentrate on ambiguous queries), Anthropic
  and OpenAI-compatible teachers with an append-only cache, prompt builder, prices
- `guarded_distillation/student.py` – TF-IDF + LinearSVC students, temperature-scaled, full-universe probs,
  OOV novelty
- `guarded_distillation/space.py` – map space (TF-IDF LSA or sentence-transformers), k-means cells, kernel W
- `guarded_distillation/gate.py` – doubt percentile, leak bound (Wilson), tonic/guard/local field, `aci_update`
- `guarded_distillation/engine.py` – the replay loop: gates, escalation, audits, cortisol updates, evidence-
  triggered retraining with shadow promotion, events
- `guarded_distillation/report.py` + `templates/replay.html` – REPORT.md, results.json, replay.html
- `guarded_distillation/cli.py` – `gdistill data | label | replay | report | demo`
- `docs/` – THEORY (lessons + guarantee), DESIGN (why it is built this way), RESULTS, ROADMAP

## Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Homebrew Python 3.11, same as ../vaers
pip install -e ".[dev]"            # add ,llm for real teachers; ,sbert for --map sbert
python -m pytest -q
gdistill demo                      # data + simulated labels + all methods + report (~3 min, free)
open runs/simulated-s0/replay.html

# real teacher, local on the A5000 (vLLM or Ollama, OpenAI-compatible), free
gdistill label --teacher openai --model qwen2.5-7b-instruct --base-url http://<a5000>:8000/v1 --price free --run runs/qwen7b-s0 --yes
gdistill replay --run runs/qwen7b-s0 && gdistill report --run runs/qwen7b-s0

# Claude teacher: ONLY with Sahil's explicit go-ahead for that run (see Status)
# gdistill label --teacher anthropic --model claude-haiku-5-5 --run runs/haiku-s0   # prints cost, stops without --yes
```

Engine settings override with `--config '{"alpha_intent":0.02}'` (see `EngineConfig`).

## Rules

- Accuracy is always scored against the datasets' human labels; the system only ever sees teacher
  labels. Keep that separation in any new metric.
- Never present simulated-teacher numbers as LLM results. The dashboard and REPORT.md label them;
  keep it that way.
- No Claude API calls without Sahil's explicit go-ahead for that specific run.
- Teacher calls cost money. `gdistill label` refuses without `--yes`. Never delete a run's
  `teacher_cache.jsonl`; reruns read from it for free.
- Guarantee invariants (docs/THEORY.md, claim G): the guard is only ever clipped from below (floor
  0), every other term of the level is bounded, and a level >= 1 accepts nothing. If you change
  any of these, update THEORY.md in the same commit.
- Tune on seed 0; report seeds 1 and 2. Give every baseline the same tuning effort before claiming
  a win.
- `data/raw/` and `runs/*/` are gitignored. Results that matter go into docs/RESULTS.md.
- Each method's replay takes ~40 s on a laptop CPU; a full 5-method run ~3 min.

## Focus

Goal now is to finish the project: working pipeline, real-teacher run (E1), results in
docs/RESULTS.md. No teaching mode, no exercises, no step-by-step hints; just do the work. The
research thread (localized risk-control guarantee) is in THEORY.md §5 if needed.
