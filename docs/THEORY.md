# Theory

Lessons 1 and 2 recap what was covered in chat. Sections 3–4 are what the code does and the one
guarantee it has. Section 5 is the open research question.

## 1. Lesson 1: split conformal prediction

A classifier says "non-serious, 92% confident". Models are overconfident, especially on unfamiliar
inputs, so the 92% is not a promise. Conformal prediction turns any model's scores into one.

1. Take labelled examples the model did not train on (the calibration set).
2. For each, measure how surprised the model was by the right answer:
   surprise = 1 − p(true label).
3. Find the 90th percentile of those surprises. Call it q.
4. For a new query, keep every label whose surprise would be at most q. That is a set of labels.

The new query's surprise is like one more draw from the same pile, so it lands above the 90th
percentile only about 10% of the time: the true label is missing from the set about 10% of the
time. The "same pile" assumption is called exchangeability.

A gate built on it: one label in the set, accept the cheap model; two or more, ask the LLM.

Why it broke on VAERS at COVID: the new reports were no longer from the same pile. The model
stayed confident while being wrong, so the sets stayed small and the gate kept accepting. The
lesson to keep: **you only learn the surprise after you find out the true answer.** Anything that
fixes drift is fed by labels that arrive later, which here means escalations and audits.

## 2. Lesson 2: moving q online (adaptive conformal inference)

Make q move with feedback. Each time a label arrives:

- **miss** (true label not in the set): raise q by (1 − α) × step
- **hit**: lower q by α × step

With α = 0.1 that is +0.9 × step and −0.1 × step. They balance exactly when misses run at 10%
(10 × 0.9 = 90 × 0.1). Above 10%, q climbs; below, it falls. This is one step of gradient descent
on the pinball (quantile) loss, which is why the q it settles on is a percentile.

**No "same pile" assumption needed.** If misses stayed above α, q would keep climbing; but once
q is high enough that every label is in the set, a miss is impossible, so q cannot climb forever.
Summing the updates: q_end − q_start = step × Σ(miss − α), so the average miss rate is within
(range of q)/(step × T) of α, for any sequence of data, including COVID.

The price: it is a long-run average. A bad month can hide inside a good year. The step size
trades reaction speed against jitter.

`guarded_distillation.gate.aci_update` is this rule, in one line; the spec is tests/test_aci.py.

## 3. What the gate controls here: leaks

A miss on an escalated query costs nothing, because the LLM answered. So each gate controls the
**leak rate**: queries where the compiled model's answer was used and it disagreed with the
teacher. That is the same update rule with "miss on an accepted query" as the error (online risk
control). Controlling plain coverage instead forced this pipeline to escalate almost everything:
with 77 intents, covering the true label 97% of the time needs large sets.

Feedback is partial. Escalated queries are labelled for sure (loss 0, known). An accepted query is
checked only if audited, with probability p chosen before its label is seen. Each step uses

    err_hat = leak × audited / p

which has expected value exactly `leak`, whatever chose p (the audit coin is independent of the
label). Unaudited accepted queries contribute err_hat = 0.

## 4. The level and its one guarantee

Doubt u(x) is the percentile of the runner-up probability among recent traffic. The gate accepts
when u(x) < 1 − level(x), with

    level(x) = tonic + guard + local[cell(x)] + κ · novelty(x)

| term | moves on | job |
|---|---|---|
| tonic | every 25 queries, from a 3,000-query window | lowest level whose leak rate is ≤ α with 95% confidence (Wilson bound over all teacher labels in the window). Conformal-PID calls this the scorecaster. |
| guard | every query, err_hat | lesson 2's rule, floored at 0: adds caution when realised leaks run above α |
| local | every teacher label in a cell, spread by kernel W, decays | how much more often the student is wrong in this region than its doubt predicts |
| novelty | nothing (label-free) | share of unseen words, distance to training queries; only adds caution |

**Claim G.** Suppose tonic ∈ [0, 1], local ∈ [−0.1, 1], κ·novelty ≥ 0, audits use p_t ≥ p_min,
and the guard follows g ← max(0, g + γ(err_hat − α)). Then for any data sequence

    (1/T) Σ err_hat_t ≤ α + (1.1 + γ/p_min) / (γ T)

and, with probability at least 1 − δ, the true average leak rate is at most that plus
sqrt(2 log(1/δ) / T) / p_min.

*Proof sketch.*
1. *The guard is bounded above.* If g ≥ 1.1 then level ≥ 1 everywhere (the other terms add at
   least −0.1), nothing is accepted, err_hat = 0, and g falls. One step adds at most γ/p_min, so
   g ≤ 1.1 + γ/p_min always.
2. *The floor only helps.* Clipping from below makes g_{t+1} ≥ g_t + γ(err_hat_t − α). Summing,
   γ Σ(err_hat − α) ≤ g_{T+1} − g_1 ≤ 1.1 + γ/p_min. Divide by γT.
3. *From estimate to truth.* M_T = Σ(leak_t − err_hat_t) is a martingale whose steps are at most
   1/p_min in size. Azuma's inequality bounds M_T / T.

Notes:
- Nothing is assumed about the data, the tonic or the local term beyond boundedness. That is why
  the tonic and local parts are free to be heuristics: the guard is the safety net.
- The 1/p_min is the cost of partial feedback: fewer audits, looser guarantee.
- It is teacher-relative (a leak is disagreement with the LLM) and a long-run average. A launch can
  push leaks above α for a while; the bound says the average catches up.
- tests/test_guarantee.py checks it by simulation under regime shifts with a useless tonic.

## 5. The research thread

The NeurIPS-shaped question: **a localized guarantee when the controller chooses its own evidence.**

- Adaptive conformal inference (Gibbs & Candès, NeurIPS 2021; JMLR 2024 for adaptive step sizes)
  and localized adaptive risk control (L-ARC: Zecchin & Simeone, NeurIPS 2024, arXiv 2405.07976)
  assume the label is revealed after every decision.
- Here the label is revealed only on escalation or audit, and the field itself sets the audit
  probabilities. Claim G covers the global average. The open part is a per-region (kernel-
  weighted) version with field-chosen p_t, and the audit allocation that minimises the bound for a
  fixed audit budget.
- Nearby work to check carefully before claiming novelty: online conformal prediction with
  bandit or semi-bandit feedback (e.g. ICLR 2026, arXiv 2604.17984), online selective conformal
  inference (Humbert et al., arXiv 2508.10336), CAP for online selective prediction (Bao et al.,
  JMLR 2025), prediction-powered inference, and active learning for risk estimation.
