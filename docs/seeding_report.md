
### Call budget 50% (gate setting chosen on seeds 0-2 for each arm; results on seeds 3-8, paired bootstrap over queries)


**Seeded vs no seeding (probation on)**

| Stream | Launch questions | First-600 launch accuracy: seeded / control | Difference, pts [95% CI] | LLM calls on those queries: arm / control | Overall accuracy diff, pts [95% CI] | Overall calls: arm / control |
|---|---|---|---|---|---|---|
| heldout | near (n=397) | 77.6% / 77.3% | +0.25 [-0.50, +1.26] (2/6 seeds) | 97% / 97% (-0.3 [-2.3, +1.8] (2/6 seeds)) | -0.01 [-0.08, +0.05] (2/6 seeds) | 54.9% / 55.2% |
| heldout | far (n=327) | 89.9% / 89.6% | +0.31 [+0.00, +0.92] (1/6 seeds) | 93% / 98% (-4.6 [-7.3, -1.8] (1/6 seeds)) | -0.01 [-0.08, +0.05] (2/6 seeds) | 54.9% / 55.2% |
| credit_cards | near domain (n=724) | 76.9% / 74.0% | +2.90 [+1.80, +4.14] (6/6 seeds) | 90% / 94% (-4.3 [-6.8, -1.8] (1/6 seeds)) | +0.20 [+0.10, +0.29] (6/6 seeds) | 50.1% / 51.0% |
| travel | far domain (n=724) | 92.3% / 88.0% | +4.28 [+2.90, +5.80] (4/6 seeds) | 97% / 96% (+1.5 [-0.4, +3.3] (4/6 seeds)) | +0.18 [+0.08, +0.28] (5/6 seeds) | 50.5% / 50.7% |

**Declared with probation only vs no seeding**

| Stream | Launch questions | First-600 launch accuracy: seeded / control | Difference, pts [95% CI] | LLM calls on those queries: arm / control | Overall accuracy diff, pts [95% CI] | Overall calls: arm / control |
|---|---|---|---|---|---|---|
| heldout | near (n=397) | 77.6% / 77.3% | +0.25 [-0.50, +1.26] (2/6 seeds) | 99% / 97% (+2.0 [+0.5, +3.8] (4/6 seeds)) | +0.01 [-0.05, +0.07] (3/6 seeds) | 54.8% / 55.2% |
| heldout | far (n=327) | 89.9% / 89.6% | +0.31 [+0.00, +0.92] (1/6 seeds) | 99% / 98% (+1.2 [-0.3, +3.1] (3/6 seeds)) | +0.01 [-0.05, +0.07] (3/6 seeds) | 54.8% / 55.2% |
| credit_cards | near domain (n=724) | 75.6% / 74.0% | +1.52 [+0.41, +2.62] (4/6 seeds) | 96% / 94% (+1.5 [-0.1, +3.2] (3/6 seeds)) | +0.09 [+0.01, +0.18] (4/6 seeds) | 50.3% / 51.0% |
| travel | far domain (n=724) | 87.8% / 88.0% | -0.14 [-0.41, +0.00] (0/6 seeds) | 96% / 96% (-0.3 [-1.0, +0.4] (0/6 seeds)) | -0.01 [-0.05, +0.03] (1/6 seeds) | 50.7% / 50.7% |

**Seeded vs no seeding, no probation**

| Stream | Launch questions | First-600 launch accuracy: seeded / control | Difference, pts [95% CI] | LLM calls on those queries: arm / control | Overall accuracy diff, pts [95% CI] | Overall calls: arm / control |
|---|---|---|---|---|---|---|
| heldout | near (n=397) | 77.6% / 76.3% | +1.26 [+0.25, +2.52] (2/6 seeds) | 92% / 95% (-3.0 [-5.8, -0.3] (1/6 seeds)) | -0.01 [-0.05, +0.03] (3/6 seeds) | 55.5% / 55.4% |
| heldout | far (n=327) | 89.9% / 89.6% | +0.31 [-0.92, +1.83] (2/6 seeds) | 85% / 91% (-5.8 [-10.1, -1.2] (1/6 seeds)) | -0.01 [-0.05, +0.03] (3/6 seeds) | 55.5% / 55.4% |
| credit_cards | near domain (n=724) | 77.3% / 70.6% | +6.77 [+5.11, +8.70] (6/6 seeds) | 71% / 80% (-8.8 [-13.1, -4.7] (0/6 seeds)) | +0.27 [+0.17, +0.36] (6/6 seeds) | 49.3% / 49.3% |
| travel | far domain (n=724) | 93.4% / 87.7% | +5.66 [+4.01, +7.32] (4/6 seeds) | 86% / 94% (-7.6 [-10.6, -4.6] (1/6 seeds)) | +0.07 [-0.04, +0.17] (4/6 seeds) | 49.9% / 49.1% |

### Call budget 60% (gate setting chosen on seeds 0-2 for each arm; results on seeds 3-8, paired bootstrap over queries)


**Seeded vs no seeding (probation on)**

| Stream | Launch questions | First-600 launch accuracy: seeded / control | Difference, pts [95% CI] | LLM calls on those queries: arm / control | Overall accuracy diff, pts [95% CI] | Overall calls: arm / control |
|---|---|---|---|---|---|---|
| heldout | near (n=397) | 77.8% / 77.8% | +0.00 [+0.00, +0.00] (0/6 seeds) | 99% / 99% (-0.3 [-1.3, +0.5] (1/6 seeds)) | +0.01 [-0.02, +0.05] (3/6 seeds) | 65.8% / 65.3% |
| heldout | far (n=327) | 89.9% / 89.6% | +0.31 [+0.00, +0.92] (1/6 seeds) | 97% / 99% (-1.5 [-3.7, +0.3] (0/6 seeds)) | +0.01 [-0.02, +0.05] (3/6 seeds) | 65.8% / 65.3% |
| credit_cards | near domain (n=724) | 76.9% / 75.8% | +1.10 [+0.41, +1.93] (4/6 seeds) | 93% / 97% (-3.7 [-5.9, -1.7] (0/6 seeds)) | +0.10 [+0.04, +0.16] (5/6 seeds) | 59.6% / 59.3% |
| travel | far domain (n=724) | 91.9% / 86.2% | +5.66 [+3.87, +7.32] (4/6 seeds) | 98% / 94% (+3.7 [+1.7, +5.7] (4/6 seeds)) | +0.16 [+0.08, +0.24] (4/6 seeds) | 59.5% / 60.1% |

**Declared with probation only vs no seeding**

| Stream | Launch questions | First-600 launch accuracy: seeded / control | Difference, pts [95% CI] | LLM calls on those queries: arm / control | Overall accuracy diff, pts [95% CI] | Overall calls: arm / control |
|---|---|---|---|---|---|---|
| heldout | near (n=397) | 77.8% / 77.8% | +0.00 [+0.00, +0.00] (0/6 seeds) | 100% / 99% (+0.5 [+0.0, +1.3] (2/6 seeds)) | +0.06 [+0.01, +0.10] (2/6 seeds) | 63.8% / 65.3% |
| heldout | far (n=327) | 89.6% / 89.6% | +0.00 [+0.00, +0.00] (0/6 seeds) | 100% / 99% (+0.9 [+0.0, +2.1] (2/6 seeds)) | +0.06 [+0.01, +0.10] (2/6 seeds) | 63.8% / 65.3% |
| credit_cards | near domain (n=724) | 76.0% / 75.8% | +0.14 [+0.00, +0.41] (1/6 seeds) | 97% / 97% (+0.7 [+0.0, +1.4] (3/6 seeds)) | +0.04 [-0.01, +0.08] (6/6 seeds) | 59.4% / 59.3% |
| travel | far domain (n=724) | 86.2% / 86.2% | +0.00 [+0.00, +0.00] (0/6 seeds) | 94% / 94% (+0.1 [+0.0, +0.4] (1/6 seeds)) | +0.01 [-0.02, +0.04] (2/6 seeds) | 59.9% / 60.1% |

**Seeded vs no seeding, no probation**

| Stream | Launch questions | First-600 launch accuracy: seeded / control | Difference, pts [95% CI] | LLM calls on those queries: arm / control | Overall accuracy diff, pts [95% CI] | Overall calls: arm / control |
|---|---|---|---|---|---|---|
| heldout | near (n=397) | 77.8% / 77.6% | +0.25 [+0.00, +0.76] (1/6 seeds) | 96% / 99% (-3.0 [-5.0, -1.3] (0/6 seeds)) | +0.02 [-0.03, +0.07] (3/6 seeds) | 64.1% / 63.8% |
| heldout | far (n=327) | 89.9% / 89.3% | +0.61 [+0.00, +1.53] (1/6 seeds) | 91% / 96% (-5.5 [-9.2, -2.1] (0/6 seeds)) | +0.02 [-0.03, +0.07] (3/6 seeds) | 64.1% / 63.8% |
| credit_cards | near domain (n=724) | 77.2% / 74.0% | +3.18 [+1.93, +4.56] (6/6 seeds) | 79% / 92% (-13.5 [-17.1, -10.2] (0/6 seeds)) | +0.16 [+0.09, +0.22] (6/6 seeds) | 59.5% / 59.5% |
| travel | far domain (n=724) | 92.8% / 87.4% | +5.39 [+3.73, +7.04] (4/6 seeds) | 91% / 95% (-3.7 [-6.4, -1.2] (1/6 seeds)) | +0.06 [-0.01, +0.13] (2/6 seeds) | 59.8% / 59.0% |