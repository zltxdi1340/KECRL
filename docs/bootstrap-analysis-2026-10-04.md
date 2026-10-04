# Paired Bootstrap Diagnostic

The analyzer now emits deterministic percentile bootstrap intervals for the
per-seed paired query-success delta against baseline. It uses 2,000 resamples
with a fixed seed and reports the raw per-seed deltas alongside the interval.

For holdout v3 at prior strength `1.5`, both `method` and `ablation_skill`
have paired delta `-0.0060`, standard deviation `0.0108`, and diagnostic
bootstrap interval `[-0.0140, 0.0030]`. At strength `0.0`, the paired delta
is exactly zero in this controlled implementation. These intervals are only
small-sample diagnostics over five seeds; they are not a frozen confidence
interval or a formal hypothesis test.

Updated summaries are in
`results/holdout_prior_v3/strength_*/analysis/summary.json`.
