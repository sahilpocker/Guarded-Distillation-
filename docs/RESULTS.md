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

## Tuning `alpha_intent` on Haiku (seed 0, then seeds 1-2)

`alpha_intent` is the leak budget for in-scope answers (default 0.03). Swept over 0.03, 0.05 and
0.08 on seed 0 with the Haiku labels, all five methods each time. Cells are LLM calls / accuracy
vs LLM only:

| Method | 0.03 (default) | 0.05 | 0.08 |
|---|---|---|---|
| Static gate | 74% / +0.3 | 68% / +0.2 | 62% / −0.2 |
| Static gate + retraining | 71% / +0.4 | 67% / +0.5 | 60% / +0.4 |
| Global cortisol | 70% / +0.4 | 76% / +0.4 | 61% / +0.6 |
| Cortisol field | 65% / +0.6 | **63% / +0.6** | 73% / +0.3 |

Rule: the highest-savings value with accuracy ≥ −0.3 pts vs LLM only, applied to the Cortisol field
(the full method). All three values pass the accuracy bar, so the pick is **0.05** (fewest calls).
Seeds 1-2 at 0.05, next to the default:

| Method | LLM calls, 0.03 → 0.05 | Cheap-path errors / 1k, 0.03 → 0.05 | Δ vs LLM only, 0.05 |
|---|---|---|---|
| Static gate | 71% → 65% | 28.3 → 44.8 | −0.03 pts |
| Static gate + retraining | 69% → 66% | 23.9 → 30.9 | +0.57 pts |
| Global cortisol | 73% → 72% | 18.1 → 26.6 | +0.40 pts |
| Cortisol field | 69% → **63%** | 23.8 → 36.2 | +0.50 pts |

(Full 3-seed table at 0.05 is in the E2 section, as the TF-IDF reference.) The field saves 6 more
points of LLM calls at the same accuracy, and the extra savings are paid for in cheap-path errors
(a looser leak budget lets more disagreements through), so this is a trade, not a free gain.

Caveats on the tuning:
- The curves are not monotone (the field uses more calls at 0.08 than at 0.03; global uses more at
  0.05 than at 0.03). With one seed and three values I read that as noise in when retrains and
  audits fall, not a trend; I did not test it.
- The pick is for the field only. At each method's own best seed-0 value, the field (63%) saves
  **no more** than static + retraining (60% at 0.08) or global (61% at 0.08). So this does not
  show an advantage for the local field. Seed 0 only; the baselines were not rerun at 0.08 on
  seeds 1-2.

## E2: frozen MiniLM embeddings + logistic regression (Haiku, `alpha_intent` 0.05, 3 seeds)

`cortisol replay --map sbert` runs both the map space and the students on `all-MiniLM-L6-v2`
embeddings (frozen), with a logistic-regression head instead of TF-IDF + linear SVM. Same labels,
streams and settings as the TF-IDF table, so the two are directly comparable.

TF-IDF student (reference, α = 0.05):

| Method | Accuracy (s0 / s1 / s2) | Δ vs LLM only, mean | LLM calls, mean | Launch questions, mean | Off-topic questions, mean | Cost per 1,000 queries, mean | Cheap-path errors / 1k, mean |
|---|---|---|---|---|---|---|---|
| LLM only | 80.7 / 79.9 / 80.8 | baseline | 100% | 84.6% | 98.6% | $0.0410 | 0.0 |
| Static gate | 80.9 / 79.8 / 80.6 | −0.03 pts | 65% | 78.6% | 98.2% | $0.0267 | 44.8 |
| Static gate + retraining | 81.2 / 80.5 / 81.4 | +0.57 pts | 66% | 85.2% | 98.5% | $0.0271 | 30.9 |
| Global cortisol | 81.1 / 80.3 / 81.2 | +0.40 pts | 72% | 84.9% | 98.6% | $0.0296 | 26.6 |
| Cortisol field | 81.3 / 80.3 / 81.3 | +0.50 pts | 63% | 85.0% | 98.6% | $0.0260 | 36.2 |

MiniLM student (E2, α = 0.05):

| Method | Accuracy (s0 / s1 / s2) | Δ vs LLM only, mean | LLM calls, mean | Launch questions, mean | Off-topic questions, mean | Cost per 1,000 queries, mean | Cheap-path errors / 1k, mean |
|---|---|---|---|---|---|---|---|
| LLM only | 80.7 / 79.9 / 80.8 | baseline | 100% | 84.6% | 98.6% | $0.0410 | 0.0 |
| Static gate | 80.5 / 80.3 / 80.1 | −0.17 pts | 51% | 72.0% | 97.5% | $0.0210 | 63.5 |
| Static gate + retraining | 81.3 / 80.7 / 81.6 | +0.73 pts | 48% | 82.9% | 97.6% | $0.0196 | 56.8 |
| Global cortisol | 81.0 / 80.4 / 81.2 | +0.40 pts | 60% | 83.1% | 98.0% | $0.0247 | 38.7 |
| Cortisol field | 80.9 / 80.4 / 81.4 | +0.43 pts | 63% | 83.6% | 98.5% | $0.0256 | 37.5 |

Reading (the roadmap expected "a larger accepted share at the same leak budget"; the result is
mixed):

1. **The embedding student answers more, but leaks more.** The ungoverned gates drop from 65-66% to
   48-51% LLM calls, and their cheap-path errors rise from 31-45 to 57-64 per 1,000. Accuracy
   against the human labels holds for retraining (+0.73 pts) and is slightly lower for the static
   gate (−0.17 pts), with a weaker launch result (72.0% vs 78.6%).
2. **The cortisol methods give the savings back for safety.** Global cortisol goes from 72% to 60%
   LLM calls, at 38.7 errors per 1,000 (up from 26.6). The field is unchanged at 63% calls and 37.5
   errors (36.2 before). Neither uses fewer calls than static + retraining (48%) in this setup, but
   both hold errors well below it (37-39 vs 57).
3. **So the clean E2 claim ("same leak budget, more answered") is only partly supported:** it holds
   for global cortisol (12 fewer points of calls, at more errors) and not for the field. I cannot
   say how much comes from the embedding student versus the embedding map, because `--map sbert`
   changed both; `--student tfidf --map sbert` and the reverse would separate them (not run).
4. **Cost of running it.** Encoding the 12,000 stream texts took about 45 s with three replays
   running at once; encoding the whole 14,272-text pool alone takes 14 s (about 1 ms per text,
   batched, on this laptop CPU). With the encodings cached, a method's replay took 2-24 s against 5-66 s for
   TF-IDF, since the dense logistic regression refits faster than the sparse n-gram SVM. In
   production the encoder would run on every query, including escalated ones.
5. No local-field advantage here either: field vs global is 63% vs 60% calls with 37.5 vs 38.7
   errors.

## Operating points: choosing the default (Haiku)

Grid on seeds 0-2: method {static + retraining, global cortisol + retraining, cortisol field} ×
`alpha_intent` {0.03, 0.05, 0.08} × student {TF-IDF, MiniLM}, TF-IDF map throughout so only the
student changes. All Haiku labels are cached, so every cell is free. "Δ" is accuracy vs LLM only
in points; ranges are the lowest and highest of the three seeds. **Launch Δ** is accuracy on the
launch-phase questions (the new product's intents, about 540 live queries per seed), the drift the
system exists to survive.

![LLM calls against launch-question accuracy vs LLM only, one point per setting](operating-points.svg)

| Method | Student | α | LLM calls, mean [range] | Launch Δ vs LLM only, pts, mean [range] | Overall Δ, pts, mean [range] | Cheap-path errors / 1k, mean [range] | Passes both bars on every seed | On curve |
|---|---|---|---|---|---|---|---|---|
| Static gate + retraining | MiniLM | 0.08 | 40% [31–53] | -5.61 [-7.42–-2.24] | 0.62 [0.49–0.74] | 90.3 [82.9–104.0] | no | yes |
| Static gate + retraining | MiniLM | 0.05 | 48% [41–60] | -2.94 [-5.94–-0.56] | 0.57 [0.25–0.89] | 58.4 [52.1–66.5] | no | yes |
| Static gate + retraining | MiniLM | 0.03 | 54% [47–64] | -0.91 [-1.62–0.37] | 0.57 [0.46–0.68] | 38.7 [34.4–43.9] | yes | yes **← default** |
| Global cortisol + retraining | MiniLM | 0.08 | 58% [55–60] | -2.99 [-3.77–-2.43] | 0.37 [0.20–0.45] | 58.7 [56.4–62.8] | no | no |
| Cortisol field | MiniLM | 0.08 | 58% [55–62] | -1.96 [-3.15–-1.12] | 0.44 [0.26–0.58] | 56.0 [54.0–57.8] | no | no |
| Static gate + retraining | TF-IDF | 0.08 | 60% [53–65] | 0.12 [-0.37–0.93] | 0.52 [0.40–0.71] | 46.7 [43.8–49.9] | yes | yes |
| Cortisol field | MiniLM | 0.05 | 60% [57–63] | -0.73 [-2.04–0.56] | 0.52 [0.26–0.76] | 42.8 [41.6–43.8] | no | no |
| Global cortisol + retraining | TF-IDF | 0.08 | 61% [61–62] | -0.97 [-1.80–0.19] | 0.47 [0.30–0.61] | 47.9 [43.2–52.6] | yes | no |
| Global cortisol + retraining | MiniLM | 0.05 | 62% [56–68] | -1.60 [-2.05–-1.30] | 0.38 [0.16–0.54] | 39.2 [33.2–45.5] | no | no |
| Cortisol field | TF-IDF | 0.05 | 63% [63–64] | 0.37 [0.19–0.56] | 0.51 [0.42–0.59] | 36.2 [34.2–39.6] | yes | yes |
| Cortisol field | TF-IDF | 0.08 | 65% [61–73] | -0.93 [-2.05–-0.36] | 0.43 [0.27–0.51] | 44.3 [33.9–51.8] | no | no |
| Global cortisol + retraining | MiniLM | 0.03 | 65% [65–66] | -0.61 [-0.72–-0.56] | 0.30 [0.30–0.30] | 24.8 [22.1–27.6] | yes | no |
| Static gate + retraining | TF-IDF | 0.05 | 66% [61–71] | 0.62 [0.00–1.11] | 0.57 [0.46–0.65] | 30.9 [29.6–33.0] | yes | yes |
| Cortisol field | MiniLM | 0.03 | 66% [66–67] | -0.12 [-0.56–0.56] | 0.36 [0.27–0.42] | 25.2 [23.1–27.0] | yes | no |
| Static gate + retraining | TF-IDF | 0.03 | 69% [62–73] | 0.68 [0.36–0.93] | 0.55 [0.45–0.69] | 23.9 [22.6–24.8] | yes | yes |
| Cortisol field | TF-IDF | 0.03 | 69% [65–74] | 0.49 [0.19–0.93] | 0.47 [0.30–0.57] | 23.8 [18.1–27.6] | yes | no |
| Global cortisol + retraining | TF-IDF | 0.05 | 72% [69–76] | 0.37 [0.19–0.54] | 0.40 [0.36–0.46] | 26.6 [21.0–34.9] | yes | no |
| Global cortisol + retraining | TF-IDF | 0.03 | 73% [70–78] | 0.19 [-0.18–0.75] | 0.36 [0.22–0.46] | 18.1 [14.5–21.2] | yes | no |

Rule for the default (fixed before running; it replaces my first rule, which used overall accuracy
alone and never bound): the setting with the fewest LLM calls such that, on every one of seeds
0-2, overall accuracy is ≥ −0.3 pts **and** launch-question accuracy is ≥ −2 pts vs LLM only.
"On curve" means no other setting has both fewer calls and higher *launch* accuracy.

- **Overall accuracy hid the drift cost.** My first pick (global cortisol + retraining, MiniLM,
  α 0.08) had launch accuracy −3.0 pts [−3.8, −2.4] while overall accuracy was above 0. It fails
  the launch bar, as do the three cheapest settings.
- **Default: static gate + retraining, MiniLM student, α 0.03** (54% LLM calls, launch −0.91 pts,
  overall +0.57 pts, 39 cheap-path errors per 1,000). Eleven of 18 settings pass both bars.
- **The cortisol field is on the curve at exactly one point** (TF-IDF, α 0.05: 63% calls, launch
  +0.37 pts). It does not use fewer calls than the default and is not the default. Beyond that
  one point it never beats the static gate with retraining at the same student and α. It stays
  experimental until E3/E4 show an advantage.
- **The tooltip carries the rest:** each point's overall Δ, ranges and cheap-path errors.
- **The best settings are not at a grid edge now (α 0.03 is the smallest)**, but the rule favours
  low α with MiniLM, and that is where the grid is thinnest.

Held-out check: the default on seeds 3-5, which were not used for any choice.

| Seed | Haiku accuracy | Default accuracy | Overall Δ | Launch Δ | LLM calls | Cheap-path errors / 1k |
|---|---|---|---|---|---|---|
| 3 | 81.3% | 81.8% | +0.49 pts | −2.38 pts | 46.2% | 48.1 |
| 4 | 80.7% | 81.1% | +0.44 pts | −1.52 pts | 58.3% | 38.9 |
| 5 | 80.5% | 81.0% | +0.41 pts | −6.32 pts | 48.0% | 42.6 |
| mean [range] | 80.8% | 81.3% | +0.45 [0.41–0.49] | −3.41 [−6.32 to −1.52] | 50.9% [46.2–58.3] | 43.2 [38.9–48.1] |

**The held-out check does not confirm the pick.** Overall accuracy and calls generalise (about
half the LLM calls, +0.45 pts, cost $0.021 against $0.041 per 1,000). Launch accuracy does not:
−3.4 pts on average, and two of three held-out seeds miss the −2 pt bar (seed 5 by a wide margin;
launch accuracy 79.3% against 85.6% for LLM only). The selection seeds gave −0.91 [−1.62, +0.37],
so the choice was within noise of the bar and likely selected on luck: each seed has about 540
launch queries, so a launch accuracy near 85% carries a sampling error of about ±1.5 pts. Off-topic accuracy is
98.0% vs 98.3%. I have not re-picked on seeds 3-5. The settings that held launch accuracy near or
above 0 on seeds 0-2 are the TF-IDF ones (for example static gate + retraining, TF-IDF, α 0.05:
66% calls, launch +0.62); they were not run on seeds 3-5.

Caveats: one real LLM, one prompt, one domain; seeds share most of the pool and are not
independent; the default needs the `[sbert]` extra (torch and a 90 MB encoder), so it is applied
with `cortisol replay --preset recommended` and `EngineConfig`'s own defaults stay light.
Reproduce: `python scripts/grid.py`, `python scripts/operating_points.py`, then
`python scripts/grid.py --seeds 3 4 5 --students minilm --alphas 0.03 --methods llm_only,retrain`
and `python scripts/operating_points.py --confirm retrain minilm 0.03 --seeds 3 4 5`.

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
