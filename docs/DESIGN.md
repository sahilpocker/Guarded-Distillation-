# Design

## The pipeline

```
query ──► scope gate ──accept "off-topic"──────────────────────────────► answer
             │ accept "banking"
             ▼
          intent gate ──accept──► intent ──► route + urgency (T0 lookups) ► answer
             │ doubt                                   │
             ▼                                         ▼ audit with probability p
          LLM (teacher) ──► answer + label ──► training pool ──► retrain (shadow, promote)
                                   │
                                   ▼
                    cortisol: tonic, guard, local cells, novelty
```

Every LLM call (escalation or audit) returns one label used by both gates. Student cost is ~0.

## Three loops, measured in evidence rather than time

| loop | trigger | acts on |
|---|---|---|
| fast | every query | guard; local term on each teacher label; decay |
| medium | every 25 queries | tonic from the 3,000-query window; calibration curve for the local term |
| slow | 400 new labels, or a new intent seen 8 times, or (full system) a hot cell plus 60 fresh labels | retrain both students; promote only if the shadow does at least as well on recent held-out labels or knows intents the current one lacks |

## Decisions and why (each one fixed a failure seen while building)

1. **Control leaks, not coverage.** Coverage at 97% over 77 intents needs big prediction sets, so
   the gate escalated almost everything. Only accepted-and-wrong costs anything.
2. **Doubt as a percentile of recent traffic.** A 77-way softmax crowds runner-up probabilities
   near 0, so a bar in probability space made the level hypersensitive near 1 (a 0.04 step swung
   acceptance by tens of points). With percentiles, level ≈ share of traffic escalated, so step
   sizes mean the same thing everywhere and across retrains.
3. **Tonic uses an upper confidence bound.** Picking the lowest level from a noisy importance-
   weighted point estimate under-escalated: a window often held only one or two audited leaks.
   The Wilson bound treats little evidence as risk.
4. **Guard floored at 0.** Unfloored, the guard drifted down between rare audited leaks and spent
   the tonic's safety margin (the level went below 0 during the launch). The floor keeps Claim G.
5. **Local term driven by residuals, using escalations too.** Raw local error rates run away: a hot
   cell escalates more, escalated queries are the doubtful ones, which are wrong more often, which
   heats the cell further. Comparing to the error rate expected at that doubt level removes the
   selection effect and isolates "confidently wrong here", which is what a launch or a policy
   change looks like.
6. **Shadow-mode history.** 77 intents from 600 warmup labels gave a 50%-accurate student. A real
   pilot starts from the client's existing agent logs anyway; 4,000 is a few days of a small desk.
7. **Simulated teacher errs on ambiguous queries.** Uniform random errors made 7% of even trivial
   queries disagree with the teacher, which capped acceptance near 25% for any student. Real LLM
   errors cluster where intents overlap; the simulator uses a cross-fitted classifier to put them
   there.
8. **Shadow test set is a random slice of recent labels.** Using the newest 15% kept the new
   intent's examples out of training, the shadow never learned it, got rejected, and the trigger
   fired again every query (a retrain storm).
9. **Novelty = unseen-word share, max'd with nearest-neighbour distance.** LSA-space distance alone
   rated "what's the weather in Paris" as ordinary banking. Word overlap with the student's own
   vocabulary catches it. With `--map sbert` the neighbour distance becomes useful on its own.

## Parameters (EngineConfig defaults)

| name | default | meaning |
|---|---|---|
| alpha_scope / alpha_intent | 0.01 / 0.03 | leak budget per query / per in-scope query (vs the teacher) |
| gamma_guard / gamma_local | 0.002 / 0.03 | step sizes |
| decay | 0.002 | per-query fade of local stress |
| novelty_gain | 0.5 | level added at novelty 1 |
| refresh / window | 25 / 3000 | tonic cadence and look-back |
| audit_base / audit_max / audit_gain | 0.08 / 0.5 / 0.5 | audit probability; directed audits add gain × local stress |
| retrain_every / novel_min / hot_threshold / hot_min_labels | 400 / 8 / 0.25 / 60 | retraining triggers |
| n_cells | 40 | map-space cells |

## Methods compared

| method | tonic | guard | local + novelty + directed audits | retraining |
|---|---|---|---|---|
| LLM only | – | – | – | – |
| Static gate | frozen at go-live | – | – | – |
| Static gate + retraining | frozen | – | – | yes |
| Global cortisol | yes | yes | – | yes |
| Cortisol field | yes | yes | yes | yes, plus hot-cell trigger |
