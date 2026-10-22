# cortisol-mvp

Sahil's product MVP: an agent pipeline that compiles its own repeated LLM decisions into cheap
CPU models and keeps them honest without anyone watching. A **cortisol field** (global + local
uncertainty level) decides, query by query, whether a compiled model may answer or the LLM must,
where to audit, and when to retrain. Demo domain: banking-support triage on Banking77 + CLINC150.

Sister project: `../vaers` (same idea on VAERS pharmacovigilance; its week-1 lesson was
"retraining is what restores accuracy under drift; gating alone does not". This repo confirms it.)

## Status (22 Oct 2026)

- Engine, CLI, tests and the replay dashboard work end to end with a **simulated** teacher.
- `cortisol.gate.aci_update` is a **deliberate stub**: Sahil writes it by hand (lesson 2 in
  docs/THEORY.md, spec in tests/test_aci_spec.py). Until then the gated methods raise
  NotImplementedError and their tests skip. **Do not implement it, paste it, or hint the formula
  in code.** If he asks for help, give hints and point at the spec, one step at a time.
- No real-LLM run yet. Next step is E1 in docs/ROADMAP.md, with a **local** teacher on the A5000.
- **The Claude API is off the table for now.** Do not make Anthropic calls, and do not touch the key in
  `../vaers/.env`, unless Sahil gives an explicit go-ahead for that specific run.

### Results so far (simulated teacher, 3 seeds, 8,000 live queries each)

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

- `cortisol/data.py` – downloads (pinned SHA-256), stream builder: shadow-mode history, then
  phases steady → launch (virtual cards + wallet pay intents appear) → oos_surge → settle
- `cortisol/taxonomy.py` – labels, launch intents, T0 routing and urgency tables
- `cortisol/teacher.py` – simulated teacher (errors concentrate on ambiguous queries), Anthropic
  and OpenAI-compatible teachers with an append-only cache, prompt builder, prices
- `cortisol/student.py` – TF-IDF + LinearSVC students, temperature-scaled, full-universe probs,
  OOV novelty
- `cortisol/space.py` – map space (TF-IDF LSA or sentence-transformers), k-means cells, kernel W
- `cortisol/gate.py` – doubt percentile, leak bound (Wilson), tonic/guard/local field, `aci_update`
- `cortisol/engine.py` – the replay loop: gates, escalation, audits, cortisol updates, evidence-
  triggered retraining with shadow promotion, events
- `cortisol/report.py` + `templates/replay.html` – REPORT.md, results.json, replay.html
- `cortisol/cli.py` – `cortisol data | label | replay | report | demo`
- `docs/` – THEORY (lessons + guarantee), DESIGN (why it is built this way), RESULTS, ROADMAP

## Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Homebrew Python 3.11, same as ../vaers
pip install -e ".[dev]"            # add ,llm for real teachers; ,sbert for --map sbert
python -m pytest -q                # 7 spec tests fail until aci_update exists; the rest pass or skip
cortisol demo                      # data + simulated labels + all methods + report (~3 min, free)
open runs/simulated-s0/replay.html

# real teacher, local on the A5000 (vLLM or Ollama, OpenAI-compatible), free
cortisol label --teacher openai --model qwen2.5-7b-instruct --base-url http://<a5000>:8000/v1 --price free --run runs/qwen7b-s0 --yes
cortisol replay --run runs/qwen7b-s0 && cortisol report --run runs/qwen7b-s0

# Claude teacher: ONLY with Sahil's explicit go-ahead for that run (see Status)
# cortisol label --teacher anthropic --model claude-haiku-5-5 --run runs/haiku-s0   # prints cost, stops without --yes
```

Engine settings override with `--config '{"alpha_intent":0.02}'` (see `EngineConfig`).

## Rules

- Accuracy is always scored against the datasets' human labels; the system only ever sees teacher
  labels. Keep that separation in any new metric.
- Never present simulated-teacher numbers as LLM results. The dashboard and REPORT.md label them;
  keep it that way.
- No Claude API calls without Sahil's explicit go-ahead for that specific run.
- Teacher calls cost money. `cortisol label` refuses without `--yes`. Never delete a run's
  `teacher_cache.jsonl`; reruns read from it for free.
- Guarantee invariants (docs/THEORY.md, claim G): the guard is only ever clipped from below (floor
  0), every other term of the level is bounded, and a level >= 1 accepts nothing. If you change
  any of these, update THEORY.md in the same commit.
- Tune on seed 0; report seeds 1 and 2. Give every baseline the same tuning effort before claiming
  a win.
- `data/raw/` and `runs/*/` are gitignored. Results that matter go into docs/RESULTS.md.
- Each method's replay takes ~40 s on a laptop CPU; a full 5-method run ~3 min.

## Teaching

Sahil is learning conformal prediction from scratch (lessons 1–2 done in chat; recap in
docs/THEORY.md). When he asks, teach from zero with this repo's examples (the banking queries, the
launch), one idea per step, and check understanding with one question. Do not skip steps or lean
on jargon; earlier, a jump straight to Azuma's inequality lost him. The research thread (a
localized risk-control guarantee when the controller chooses its own audits) is in THEORY.md §5.
