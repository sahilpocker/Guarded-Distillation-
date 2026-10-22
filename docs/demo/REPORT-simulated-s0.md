# Replay report: simulated-s0

Teacher: **simulated (acc 93%)**. Stream: 8,000 live queries after 4,000 shadow-mode queries (Banking77 + CLINC150 out-of-scope, seed 0). Accuracy is against the datasets' human labels; the system itself only ever sees the teacher's labels.

> The teacher is SIMULATED (gold label, wrong mostly on ambiguous queries). These numbers show the mechanism; they are not a measurement of any real LLM.

| Method | Accuracy | vs LLM only | LLM calls | Cost / 1k queries | Cheap-path errors / 1k | Launch queries | Off-topic queries | Retrains |
|---|---|---|---|---|---|---|---|---|
| LLM only | 92.8% | baseline | 100% | $0.0268 | 0.0 | 94.8% | 97.6% | 0 |
| Static gate | 91.4% | -1.5 pts | 45% | $0.0120 | 29.8 | 80.9% | 95.9% | 0 |
| Static gate + retraining | 92.5% | -0.3 pts | 41% | $0.0111 | 18.0 | 92.2% | 96.8% | 4 |
| Global cortisol | 92.7% | -0.2 pts | 49% | $0.0133 | 11.8 | 92.8% | 96.8% | 4 |
| Cortisol field | 92.5% | -0.3 pts | 47% | $0.0127 | 12.8 | 91.7% | 97.3% | 4 |

## By phase (accuracy / LLM calls)

| Method | steady | launch | oos_surge | settle |
|---|---|---|---|---|
| LLM only | 92.3% / 100% | 92.9% / 100% | 93.6% / 100% | 93.2% / 100% |
| Static gate | 91.8% / 41% | 90.5% / 47% | 92.2% / 49% | 91.2% / 45% |
| Static gate + retraining | 91.9% / 41% | 92.6% / 47% | 93.4% / 52% | 92.8% / 24% |
| Global cortisol | 92.1% / 64% | 92.9% / 45% | 93.4% / 51% | 92.7% / 31% |
| Cortisol field | 91.9% / 53% | 92.5% / 44% | 93.5% / 50% | 92.9% / 40% |

Cheap-path errors: answers the student gave (no LLM) that disagree with the human label. Cost uses the price profile in results.json; onboarding (shadow mode) cost is reported separately there.
