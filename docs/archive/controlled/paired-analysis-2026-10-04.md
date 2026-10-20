# Paired Seed Analysis

The stage analyzer now reports per-seed paired query-success deltas against
the baseline, along with standard deviations for each support checkpoint.
Pairing uses the same seed and manifest role split; it does not pool query
episodes across variants.

For holdout v3 at prior strength `1.5`, the paired query-success delta was
`-0.0060 ± 0.0108` for both `method` and `ablation_skill`. At strength `0.0`,
the paired delta was `0.0` for all variants because the controlled policy path
is identical when the prior is disabled. These are diagnostic summaries over
five seeds and remain `formal_result=false`.

The analyzer output is in
`results/holdout_prior_v3/strength_*/analysis/summary.json`. The reported
standard deviations describe seed variability; confidence intervals and any
formal hypothesis procedure remain to be frozen before paper experiments.
