# Replay report: grid-minilm-a03-s0

Teacher: **anthropic:claude-haiku-5-5**. Stream: 8,000 live queries after 4,000 shadow-mode queries (Banking77 + CLINC150 out-of-scope, seed 0). Accuracy is against the datasets' human labels; the system itself only ever sees the teacher's labels.

| Method | Accuracy | vs LLM only | LLM calls | Cost / 1k queries | Cheap-path errors / 1k | Launch queries | Off-topic queries | Retrains |
|---|---|---|---|---|---|---|---|---|
| LLM only | 80.7% | baseline | 100% | $0.0410 | 0.0 | 84.6% | 98.4% | 0 |
| Static gate | 80.6% | -0.1 pts | 53% | $0.0216 | 43.4 | 75.1% | 98.1% | 0 |
| Static gate + retraining | 81.2% | +0.5 pts | 47% | $0.0195 | 43.9 | 83.1% | 96.8% | 4 |
| Global cortisol | 81.0% | +0.3 pts | 65% | $0.0267 | 24.8 | 84.0% | 98.1% | 4 |
| Cortisol field | 81.0% | +0.3 pts | 67% | $0.0276 | 23.1 | 84.0% | 98.1% | 5 |

## By phase (accuracy / LLM calls)

| Method | steady | launch | oos_surge | settle |
|---|---|---|---|---|
| LLM only | 79.4% / 100% | 81.1% / 100% | 83.7% / 100% | 80.1% / 100% |
| Static gate | 80.2% / 49% | 80.3% / 55% | 83.8% / 56% | 79.5% / 54% |
| Static gate + retraining | 80.2% / 49% | 81.0% / 48% | 83.9% / 48% | 81.1% / 45% |
| Global cortisol | 79.9% / 70% | 81.2% / 60% | 83.7% / 66% | 80.5% / 64% |
| Cortisol field | 79.9% / 71% | 81.2% / 62% | 83.8% / 69% | 80.4% / 67% |

Cheap-path errors: answers the student gave (no LLM) that disagree with the human label. Cost uses the price profile in results.json; onboarding (shadow mode) cost is reported separately there.
