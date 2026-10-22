# Replay report: grid-minilm-a08-s0

Teacher: **anthropic:claude-haiku-5-5**. Stream: 8,000 live queries after 4,000 shadow-mode queries (Banking77 + CLINC150 out-of-scope, seed 0). Accuracy is against the datasets' human labels; the system itself only ever sees the teacher's labels.

| Method | Accuracy | vs LLM only | LLM calls | Cost / 1k queries | Cheap-path errors / 1k | Launch queries | Off-topic queries | Retrains |
|---|---|---|---|---|---|---|---|---|
| LLM only | 80.7% | baseline | 100% | $0.0410 | 0.0 | 84.6% | 98.4% | 0 |
| Static gate | 80.2% | -0.5 pts | 36% | $0.0146 | 107.4 | 66.0% | 97.0% | 0 |
| Static gate + retraining | 81.2% | +0.5 pts | 31% | $0.0128 | 104.0 | 77.2% | 97.6% | 3 |
| Global cortisol | 80.9% | +0.2 pts | 58% | $0.0236 | 57.0 | 81.8% | 98.1% | 4 |
| Cortisol field | 81.0% | +0.3 pts | 57% | $0.0232 | 57.8 | 81.4% | 98.1% | 4 |

## By phase (accuracy / LLM calls)

| Method | steady | launch | oos_surge | settle |
|---|---|---|---|---|
| LLM only | 79.4% / 100% | 81.1% / 100% | 83.7% / 100% | 80.1% / 100% |
| Static gate | 80.6% / 31% | 78.6% / 38% | 83.7% / 42% | 79.2% / 36% |
| Static gate + retraining | 80.6% / 31% | 80.2% / 32% | 84.0% / 35% | 81.4% / 26% |
| Global cortisol | 79.9% / 67% | 81.0% / 54% | 83.3% / 48% | 80.6% / 55% |
| Cortisol field | 80.0% / 66% | 81.0% / 49% | 83.4% / 55% | 80.7% / 53% |

Cheap-path errors: answers the student gave (no LLM) that disagree with the human label. Cost uses the price profile in results.json; onboarding (shadow mode) cost is reported separately there.
