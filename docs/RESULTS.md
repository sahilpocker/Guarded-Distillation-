# Results

Two teachers, same streams and settings: 4,000 shadow-mode queries then 8,000 live queries,
default EngineConfig (not re-tuned for the real teacher). Accuracy is always against the human
labels; the system only sees teacher labels.

1. **Simulated teacher** (93% accurate, errors concentrated on ambiguous queries). Shows the
   mechanism, not a real LLM.
2. **Claude Haiku 5.5** (E1, real LLM, 1-shot-per-intent prompt). Section below.

Reproduce the simulated runs: `cortisol demo --seed K --run runs/simulated-sK` for K = 0, 1, 2.

## Headline, simulated teacher (3 seeds)

| Method | Accuracy (s0 / s1 / s2) | Δ vs LLM only, mean | LLM calls, mean | Launch questions, mean | Cheap-path errors / 1k, mean |
|---|---|---|---|---|---|
| LLM only | 92.85 / 92.81 / 92.50 | baseline | 100% | 94.5% | 0 |
| Static gate | 91.37 / 90.60 / 90.46 | −1.91 pts | 46.3% | 76.1% | 34.0 |
| Static gate + retraining | 92.50 / 92.61 / 92.31 | −0.25 pts | 46.7% | 93.3% | 13.0 |
| Global cortisol | 92.66 / 92.49 / 92.24 | −0.26 pts | 48.2% | 92.9% | 13.0 |
| Cortisol field | 92.55 / 92.41 / 92.15 | −0.35 pts | 46.7% | 92.0% | 16.1 |

Cost at Claude Haiku 5.5 list prices (seed 0; token counts calibrated on the real Haiku pilot):
LLM only $0.041 per 1,000 queries, cortisol field $0.019. At these prices the dollars are small; the case for compiling is volume, latency, owning
the model, and pricier teachers.

## Headline, Claude Haiku 5.5 teacher (3 seeds)

Teacher accuracy against the human labels: **80.7 / 79.9 / 80.8%** (seeds 0 / 1 / 2) on the live
stream, versus 92.8% for the simulated teacher. Over the whole 14,194-text pool it is 81.1%:
79.5% on the 77 Banking77 intents, 98.7% on out-of-scope. The misses are near-synonym intents
(`balance_not_updated_after_bank_transfer` vs `pending_transfer` / `transfer_timing`,
`direct_debit_payment_not_recognised` vs `card_payment_not_recognised`, `beneficiary_not_allowed`
vs `declined_transfer`). The simulated teacher was too kind.

| Method | Accuracy (s0 / s1 / s2) | Δ vs LLM only, mean | LLM calls, mean | Launch questions, mean | Off-topic questions, mean | Cost per 1,000 queries, mean | Cheap-path errors / 1k, mean |
|---|---|---|---|---|---|---|---|
| LLM only | 80.7 / 79.9 / 80.8 | baseline | 100% | 84.6% | 98.6% | $0.0410 | 0.0 |
| Static gate | 81.0 / 80.0 / 80.9 | +0.17 pts | 71% | 81.3% | 98.4% | $0.0293 | 28.3 |
| Static gate + retraining | 81.2 / 80.5 / 81.4 | +0.57 pts | 69% | 85.2% | 98.6% | $0.0282 | 23.9 |
| Global cortisol | 81.1 / 80.2 / 81.2 | +0.37 pts | 73% | 84.7% | 98.6% | $0.0298 | 18.1 |
| Cortisol field | 81.3 / 80.2 / 81.3 | +0.47 pts | 69% | 85.1% | 98.6% | $0.0282 | 23.8 |

Cost is the real token usage Haiku returned (about 3,330 cached prompt tokens read per call, 24
uncached input, 10 output) at the `haiku-5.5` price profile: $0.10 / $0.50 per million input /
output tokens, cache reads $0.01. Labelling the whole pool once cost **$0.58** (14,194 calls,
$0.000041 each); every seed's stream is read from that shared cache.

Seed 0 by phase (accuracy / LLM calls):

| Method | steady | launch | off-topic surge | settle |
|---|---|---|---|---|
| LLM only | 79.4% / 100% | 81.1% / 100% | 83.7% / 100% | 80.1% / 100% |
| Static gate | 79.9% / 72% | 81.2% / 77% | 83.7% / 76% | 80.7% / 74% |
| Static gate + retraining | 79.8% / 71% | 81.5% / 74% | 83.7% / 75% | 81.0% / 63% |
| Global cortisol | 79.6% / 80% | 81.6% / 66% | 84.2% / 64% | 80.6% / 64% |
| Cortisol field | 79.8% / 78% | 81.7% / 62% | 84.7% / 55% | 80.8% / 56% |

### Reading the Haiku results

1. **Compiling still removes LLM calls, but fewer: about 27–31% instead of ~53%.** Cost falls from
   $0.041 to $0.028–0.030 per 1,000 queries. The likely reason (not tested): Haiku's labels on
   confusable intents are noisy, so a student agrees with the teacher less often at any given
   confidence, and the gate, which controls disagreement with the teacher, has to escalate more.
2. **No accuracy loss against the human labels.** Every compiled method is +0.2 to +0.6 pts above
   Haiku. That is inside the noise (one method on one seed has a standard error of about 0.45
   pts at 8,000 queries; I have not run a paired test), so read it as "no measurable loss", not
   "the student beats the teacher".
3. **Retraining still matters at the launch.** The gate that never retrains drops to 81.3% on
   launch questions (−3.3 pts vs Haiku); every retraining variant is at 84.7–85.2%, level with
   Haiku. Same direction as the simulated runs, smaller gap.
4. **The local field shows no clear advantage here either.** Field vs global: 69% vs 73% LLM
   calls, but 23.8 vs 18.1 cheap-path errors per 1,000; accuracy within noise. Mixed, so no
   claim. E3 and E4 are still the way to find out.

## Seed 0 by phase, simulated teacher (accuracy / LLM calls)

| Method | steady | launch | off-topic surge | settle |
|---|---|---|---|---|
| LLM only | 92.3% / 100% | 92.9% / 100% | 93.6% / 100% | 93.2% / 100% |
| Static gate | 91.8% / 41% | 90.5% / 47% | 92.2% / 49% | 91.2% / 45% |
| Static gate + retraining | 91.9% / 41% | 92.6% / 47% | 93.4% / 52% | 92.8% / 24% |
| Global cortisol | 92.1% / 64% | 92.9% / 45% | 93.4% / 51% | 92.7% / 31% |
| Cortisol field | 91.9% / 53% | 92.5% / 44% | 93.5% / 50% | 92.9% / 40% |

## What the simulated runs say

1. Compiling plus retraining handles about half the traffic without the LLM, within ~0.3 points of
   LLM-only accuracy, through a product launch and an off-topic surge, with no human step.
2. Retraining is what saves the launch: a gate that never retrains drops to ~76% on the new
   product's questions; every retraining variant stays at 92–93%.
3. The adaptive level (tonic + guard) keeps cheap-path errors at the retraining variant's level or
   lower; the static gate leaks almost three times as much.
4. The local field shows **no advantage on this stream**: slightly fewer LLM calls than global,
   slightly lower accuracy, within seed noise. The launch intents are learned within ~60 labels,
   so local stress has little time to matter. E3 and E4 are designed to find out where it does.

## Caveats

- The simulated numbers are a mechanism demo and flatter the real teacher (93% vs about 80%).
- One real LLM (Haiku 5.5, one example per intent, no tuning of the prompt), one domain, three
  seeds. Engine defaults were tuned lightly on the simulated seed 0 and not re-tuned for Haiku.
- Banking77's 77 intents overlap, so human labels themselves are debatable on the confusable
  pairs; "accuracy against human labels" is partly a measure of agreement with that labelling.
- The three seeds share most of the pool, so they are not independent samples.
- Haiku 5.5 prices are the figures in `cortisol.teacher.PRICES`, matching Anthropic's published
  table as cached on 2026-10-06; re-check before quoting dollars.
- Leaks are controlled relative to the teacher, so the student inherits the teacher's mistakes on
  accepted queries; accuracy is reported against human labels to keep that visible.
