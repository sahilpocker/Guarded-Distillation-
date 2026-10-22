# Roadmap

In order. Each experiment states what would count as a result before running it.

## E1. Real teacher (do first)

Label seed 0 with a local instruct model on the A5000 via vLLM or Ollama
(`--teacher openai --base-url http://<a5000>:8000/v1 --price free`), then replay all methods.
About 12,000 calls, free.
- Report the teacher's own accuracy against human labels, overall and on the launch intents.
- Same table as RESULTS.md. Then seeds 1 and 2 (stream and labels change per seed).
- A Claude teacher (Haiku 5.5) is supported but off the table for now: only with Sahil's explicit
  go-ahead for that run. Estimated under $3 for 12,000 calls at list prices.

## E2. Better student and map

`pip install -e ".[sbert]"`, `--map sbert`, and a student on frozen sentence embeddings + logistic
regression (add as `Student` backend). Expect a larger accepted share at the same leak budget.
Result: LLM calls at equal accuracy, versus TF-IDF.

## E3. Concept drift without a new label (where local should matter)

Add a phase where the teacher's answer for one existing intent changes (a policy change: e.g.
`card_arrival` questions are now handled as `card_delivery_estimate`; update gold the same way).
No new intent appears, so the novel-label trigger never fires; the student stays confidently wrong
in one region. Needs forgetting: when a hot cell's recent teacher labels contradict old pool labels
there, down-weight or drop the old ones before retraining.
Result: cheap-path errors during the drift phase, cortisol field vs global vs static + retraining,
at equal LLM calls. Local should win here; if it does not, say so.

## E4. Audit budget sweep

audit_base in {0.02, 0.05, 0.08, 0.15}. Directed audits should matter most when audits are scarce.
Result: cheap-path errors vs total LLM calls (escalations + audits), one curve per method.

## E5. Ablations of the full system

Remove one at a time: novelty, directed audits, local term, hot-cell trigger. Three seeds each.

## E6. Teacher swap

Simulated accuracy 0.85 / 0.93 / 0.98, plus real small vs large teachers. Show that the same
pipeline reaches a similar leak rate with a different structure and cost each time ("system
complexity measures model weakness").

## E7. Spike memory

A stream where a drift recurs (a seasonal intent cluster that comes back). Store each resolved
spike's signature (cells, skills, score shift) with the student version that fixed it; on a
matching spike, propose the stored fix and confirm it on a small audit batch before switching.

## E8. Does error cluster in map space?

Moran's I of student errors on the cell graph, per phase. If errors do not cluster, local cortisol
cannot beat global and E3 should be redesigned.

## Product track

- P1. `compile(skill, traces)` as a library call other agents use: returns a gated model with its
  own cortisol, audits and demotion.
- P2. "Compile audit" offer: a client shares a week of agent logs; report which steps are
  compilable, the share of calls, projected savings and the risk budget.
- P3. Pitch: the replay dashboard plus one slide per decision in docs/DESIGN.md.
- P4. A second domain with a multi-step agent where each LLM call is expensive.

## Research track

The localized guarantee with self-chosen audits (docs/THEORY.md §5). Start by writing the theorem
statement and checking the nearby literature listed there.
