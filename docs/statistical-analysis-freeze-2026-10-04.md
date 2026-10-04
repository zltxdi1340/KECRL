# Stage Statistical Analysis Freeze

`configs/statistical_analysis_v1.yaml` freezes the analysis procedure for the
next controlled comparison. The primary object is the independent query
success curve over support checkpoints `0,50,100,200`; a run below the query
threshold remains right-censored. Results are paired by seed against the
baseline and report per-seed deltas, means, and sample standard deviations.

The 2,000-resample percentile bootstrap interval is deterministic and useful
for diagnostics. It is explicitly not a formal confidence interval claim for
the paper because the current stage uses only five seeds and the controlled
environment. Multiple-comparison correction and formal inference remain
pending the final seed budget and statistical review.

This freeze defines how the next stage output is summarized. It does not
promote any existing smoke, holdout, prior scan, or controlled-stage output to
`formal_result=true`.
