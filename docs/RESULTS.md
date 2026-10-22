# Results

All runs: simulated teacher (93% accurate against human labels, errors concentrated on ambiguous
queries), 4,000 shadow-mode queries then 8,000 live queries, default EngineConfig. Accuracy is
against the human labels. **These show the mechanism, not a real LLM.** E1 replaces the teacher.

Reproduce: `cortisol demo --seed K --run runs/simulated-sK` for K = 0, 1, 2.

## Headline (3 seeds)

| Method | Accuracy (s0 / s1 / s2) | Δ vs LLM only, mean | LLM calls, mean | Launch questions, mean | Cheap-path errors / 1k, mean |
|---|---|---|---|---|---|
| LLM only | 92.85 / 92.81 / 92.50 | baseline | 100% | 94.5% | 0 |
| Static gate | 91.37 / 90.60 / 90.46 | −1.91 pts | 46.3% | 76.1% | 34.0 |
| Static gate + retraining | 92.50 / 92.61 / 92.31 | −0.25 pts | 46.7% | 93.3% | 13.0 |
| Global cortisol | 92.66 / 92.49 / 92.24 | −0.26 pts | 48.2% | 92.9% | 13.0 |
| Cortisol field | 92.55 / 92.41 / 92.15 | −0.35 pts | 46.7% | 92.0% | 16.1 |

Cost at Claude Haiku 5.5 list prices (seed 0): LLM only $0.027 per 1,000 queries, cortisol field
$0.013. At these prices the dollars are small; the case for compiling is volume, latency, owning
the model, and pricier teachers.

## Seed 0 by phase (accuracy / LLM calls)

| Method | steady | launch | off-topic surge | settle |
|---|---|---|---|---|
| LLM only | 92.3% / 100% | 92.9% / 100% | 93.6% / 100% | 93.2% / 100% |
| Static gate | 91.8% / 41% | 90.5% / 47% | 92.2% / 49% | 91.2% / 45% |
| Static gate + retraining | 91.9% / 41% | 92.6% / 47% | 93.4% / 52% | 92.8% / 24% |
| Global cortisol | 92.1% / 64% | 92.9% / 45% | 93.4% / 51% | 92.7% / 31% |
| Cortisol field | 91.9% / 53% | 92.5% / 44% | 93.5% / 50% | 92.9% / 40% |

## What this says

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

- Simulated teacher. Real LLM errors may be distributed differently; E1 measures that.
- One domain, three seeds, default settings tuned lightly on seed 0.
- Leaks are controlled relative to the teacher, so the student inherits the teacher's mistakes on
  accepted queries; accuracy is reported against human labels to keep that visible.
