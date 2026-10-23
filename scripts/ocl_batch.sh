#!/bin/sh
# OCL on all three families, seeds 3-5, over a grid of cost coefficients (the cliff is narrow: see docs/RESULTS.md)
OCLPY=${OCLPY:-third_party/ocl-env/bin/python}
for fam in ${FAMS:-banking credit_cards travel}; do for seed in 3 4 5; do for mu in ${MUS:-4.0 4.15 4.2 4.25 4.3 4.4}; do
  [ -f runs/ocl/$fam-s$seed-mu$mu.json ] || echo "PYTHONPATH= $OCLPY scripts/ocl_adapter.py run --family $fam --seed $seed --mu $mu > /dev/null 2>&1"
done; done; done | xargs -P ${JOBS:-5} -I{} sh -c "{}"
