# Status report (28 Oct 2026, updated at v1.0)

**Update at v1.0.** The steps in section 5 are done except the Sonnet replay, which stopped for budget. Outcomes:

1. **Site rework:** done (executive overview with savings calculator, risks and catches, comparison table, routing chart as points versus the big AI, evidence section).
2. **Cortisol question:** closed as a negative result. E3 (policy change) and E5 (ablations), six held-out seeds, paired bootstrap: the field is not better than a plain gate at the same call share, and none of its parts has an effect of its own. Per-intent stress (semantic cortisol) is the part that helps under drift. See `docs/RESULTS.md`, "Closing the cortisol question".
3. **Second teacher:** Sonnet 5.5 labelled 3,282 wordings ($1.59 spent; budget exhausted). It is 85.4% accurate against human labels vs Haiku's 81.2%. **No Sonnet replay was run**; the savings-versus-price table is arithmetic on the measured 51% cost ratio.
4. **Service:** load test (one process saturates near 270 requests/s; p95 15 ms at one request in flight, 104 ms at 16) and a security scan (no vulnerable dependencies, no secrets, findings fixed or annotated; CI job added). Tagged v1.0.

Still open: a Sonnet replay (about $7 for the pool), E4, E6 with a replayed second teacher, E7, E8, the localized guarantee, a second domain and multi-step agents (moved to a separate project), real customer logs.

---

*The original report follows, as written on 28 Oct.*

## 1. Where we started (22 Oct)

- **Idea:** a pipeline that compiles an agent's repeated LLM decisions into cheap CPU models. A "cortisol field" of
  global and local uncertainty decides per query whether the small model may answer, where to audit, and when to retrain.
- **Plan (`docs/ROADMAP.md`):**
  - Experiments: E1 real teacher; E2 better student; E3 drift without a new label (where local cortisol should win);
    E4 audit budget sweep; E5 ablations; E6 teacher swap; E7 spike memory; E8 do errors cluster spatially.
  - Product track: P1 `compile(skill, traces)` as a library; P2 a "compile audit" offer for client logs; P3 pitch;
    P4 a second, multi-step domain.
  - Research track: a localized risk-control guarantee.

## 2. What happened

Done and solid:

- **Gate and engine:** a replay engine, a conformal-style gate, and a Haiku 5.5 teacher. The whole pool is labelled for about $0.80.
- **Recommended setting:** static gate, retraining, MiniLM student and class balancing. On held-out seeds it uses about
  50% of the calls at about the same accuracy.
- **Learned router:** it beats the old margin score at matched calls. A paired bootstrap over seeds 3-8 excludes zero on
  overall accuracy in every cell.
- **Seeding:** synthetic examples (Nemotron) help whole-domain launches, with +2.9 to +4.3 points on new-product accuracy at 50% calls.
- **Failures and weak spots, all documented:** the original cortisol field never beat the plain gate, the first pick lost
  3.4 points on launches, and a whole new domain remains 2-3 points behind.
- **Ops:** CI eval gate, FastAPI service on Render, persistence across restarts (verified 28 Oct), spend cap, monitoring,
  alerts, model registry with hot-swap and rollback.
- **Site:** live, but it needs a rework (executive overview, real console, honest charts).

## 3. Gaps

1. **The original thesis is unproven.** The local cortisol field is demoted to an appendix. E3, the experiment built to
   favour it, was never run. E4, E5, E7 and E8 were also not run. We pivoted to routing and seeding, which worked, but
   that is a different paper from the one we planned.
2. **Narrow evidence.** One task type (intent classification), two datasets, one teacher, and a simulated launch built by
   holding out intents. No real logs and no multi-step agent. Everything rests on Haiku, which is only about 80% accurate
   against human labels.
3. **Savings are small in dollars.** $0.021 vs $0.041 per 1,000 questions with Haiku, which is $20 saved per million
   questions. The commercial case needs a pricier teacher or expensive multi-step calls, and we have not shown that.
4. **The guarantee is weaker than it sounds.** Leaks stay near the budget in our runs, but the claim holds under
   assumptions we have not stress-tested. The localized guarantee is unstarted.
5. **Ops layer, mild.** Free-tier host, one restart-sensitive counter, and no load numbers.
6. **The site undersells and confuses.**

## 4. Direction options

- **A. Research paper:** "learned routing under risk control with drift." Needs a second teacher, a second dataset
  family, baselines such as FrugalGPT and cascade methods, and E3/E5 to settle the cortisol question.
- **B. Product or tool:** `compile()` as a library, the "compile audit" offer, and a domain config so any task plugs in.
  This needs the generality proof first.
- **C. Portfolio piece:** finish the site, the load test and the security scan, and stop.

## 5. Decision (28 Oct)

Do C now, then close the open evidence questions, then freeze v1.0:

1. Site rework: executive overview, real console, honest charts.
2. Close the cortisol question with no API spend: E3 as a simulated policy change and E5 ablations, on seeds 3-8 with
   paired bootstrap CIs. Write the outcome plainly in RESULTS.md, whichever way it goes.
3. Sonnet 5.5 as a second teacher on the Banking77 pool, cap $10. Report how LLM-only cost and our savings change with
   teacher price.
4. Load test and security scan of the service, then freeze: tag v1.0, update the README and this file.

Out of scope for now: the domain-config refactor (the multi-step agent work moves to a separate project), Kubernetes,
canary routing.
