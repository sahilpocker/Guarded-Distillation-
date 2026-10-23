# Synthetic test data: pilot (300 requested, 290 generated, 222 kept)

**Why.** Every real-data run draws from one pool of 14,194 questions, so seeds share wording. Synthetic questions give
fresh wording and let me control what varies (style, which intent). **What it is not:** a replacement for the
human-labelled results. Synthetic text is cleaner than real traffic and the labels are the generator's intent, checked by
Haiku, not a human's. Results on it are a robustness suite, reported separately.

**How** (`scripts/synth.py`).
1. *Generate.* `nvidia/nemotron-3.5-lightning-30b-a3b` on NVIDIA's hosted API (reasoning switched off). For each of 15
   Banking77 intents, five real examples are shown and the model writes five *new* messages in each of four styles: plain
   (one clear sentence), casual (lowercase, slang, typos), context (two or three sentences of background) and terse (under
   seven words). 15 intents x 4 styles x 5 = 300 requested; 60 calls, 16,606 tokens; 290 came back (some replies held fewer
   than five). The 15 intents are the five launch intents (virtual cards, wallet pay), five card issues that overlap
   (lost, compromised, not working, arrival, delivery) and five others.
2. *Curate.* Drop exact duplicates (0), questions already in the real data (2), near-duplicates of an earlier item of the
   same intent (MiniLM cosine >= 0.92; 4), and anything Haiku, labelling blind with the same 77-intent prompt as the main
   runs, does not label as the generator's intent (63). 222 kept. Haiku cost $0.012.
3. *Probe.* Run a model bundle's gate over the kept items and score it against the intent.

**Curation numbers: Haiku agrees with the generator's intent**

| Style | Agreement |
|---|---|
| plain | 84% |
| casual | 83% |
| context | 77% |
| terse | 69% |

| Lowest-agreement intents | Agreement |
|---|---|
| compromised_card | 40% |
| card_not_working | 50% |
| apple_pay_or_google_pay | 65% |
| transfer_not_received_by_recipient | 67% |

Reading the disagreements, most are real ambiguity rather than a bad generator or a bad Haiku: "someone used my card
online without my permission" is `compromised_card` to the generator and `card_payment_not_recognised` to Haiku, and
Banking77's own labels blur the same line; "Help lost card" can be either of two intents. Terse messages are the least
identifiable, which is what you would expect.

**Probe: the gate on the 222 kept items** (accuracy of the student's top intent; "accepted" = share the gate answers itself;
"wrong" = accepted and wrong)

| Style (n) | v1: accuracy / accepted / wrong | v2: accuracy / accepted / wrong |
|---|---|---|
| plain (59) | 54% / 31% / 6.8% | 88% / 54% / 0.0% |
| casual (61) | 51% / 33% / 8.2% | 85% / 44% / 3.3% |
| context (58) | 50% / 16% / 0.0% | 78% / 36% / 3.4% |
| terse (44) | 59% / 27% / 2.3% | 89% / 34% / 0.0% |

v1 is trained on history only (it has never seen the launch intents); v2 is the registry model trained with the launch-phase
labels. Two readings, both tentative: the long, chatty style is the hardest for the student (78% against 85-89%), and the
gate accepts the fewest answers there, so it is cautious where it should be; and v2 beats v1 by about 30 points on fresh
wording, so the gain from learning the launch intents is not an artifact of the shared pool. With 44-61 items per style the
style differences are inside sampling error (a 10-point gap on n = 58 is about two standard errors at best).

**Caveats.**
- Keeping only items Haiku agreed with removes the ambiguous ones, so accuracy here is an overestimate of what real traffic
  would give.
- 15 of 77 intents, one generator, one pilot. The generator's output may be subject to the model's licence on
  build.nvidia.com; check it before redistributing the data. The files are in `data/synth/`.
- Rate limits: NVIDIA's free endpoint returned 429 with four parallel requests; the script uses two with backoff. Most of
  the catalogue's older models return 404 for this account; the working one is named in the script.

**Next, if this is worth scaling.** (1) All 77 intents with 20-40 messages each, about 3,000-6,000 items. (2) Controlled
*new* intents at set distances from existing ones, to test the adjacent-versus-distant finding directly. (3) A synthetic
stream with a controlled launch and style shift, replayed through the gate and the learned router. (4) A 100-item human
spot-check of the kept set before quoting any accuracy from it.
