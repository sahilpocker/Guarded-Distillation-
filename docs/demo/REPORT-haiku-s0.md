# Replay report: haiku-s0

Teacher: **anthropic:claude-haiku-5-5**. Stream: 8,000 live queries after 4,000 shadow-mode queries (Banking77 + CLINC150 out-of-scope, seed 0). Accuracy is against the datasets' human labels; the system itself only ever sees the teacher's labels.

| Method | Accuracy | vs LLM only | LLM calls | Cost / 1k queries | Cheap-path errors / 1k | Launch queries | Off-topic queries | Retrains |
|---|---|---|---|---|---|---|---|---|
| LLM only | 80.7% | baseline | 100% | $0.0410 | 0.0 | 84.6% | 98.4% | 0 |
| Static gate | 81.0% | +0.3 pts | 74% | $0.0305 | 25.5 | 83.9% | 98.4% | 0 |
| Static gate + retraining | 81.2% | +0.4 pts | 71% | $0.0290 | 24.2 | 85.5% | 98.4% | 5 |
| Global cortisol | 81.1% | +0.4 pts | 70% | $0.0288 | 18.6 | 84.6% | 98.4% | 5 |
| Cortisol field | 81.3% | +0.6 pts | 65% | $0.0268 | 27.6 | 85.5% | 98.4% | 4 |

## By phase (accuracy / LLM calls)

| Method | steady | launch | oos_surge | settle |
|---|---|---|---|---|
| LLM only | 79.4% / 100% | 81.1% / 100% | 83.7% / 100% | 80.1% / 100% |
| Static gate | 79.9% / 72% | 81.2% / 77% | 83.7% / 76% | 80.7% / 74% |
| Static gate + retraining | 79.8% / 71% | 81.5% / 74% | 83.7% / 75% | 81.0% / 63% |
| Global cortisol | 79.6% / 80% | 81.6% / 66% | 84.2% / 64% | 80.6% / 64% |
| Cortisol field | 79.8% / 78% | 81.7% / 62% | 84.7% / 55% | 80.8% / 56% |

Cheap-path errors: answers the student gave (no LLM) that disagree with the human label. Cost uses the price profile in results.json; onboarding (shadow mode) cost is reported separately there.
