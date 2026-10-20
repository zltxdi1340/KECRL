# Query Efficiency Metric Sensitivity

This note evaluates alternative query-success thresholds on the existing
controlled candidate run at
`results/controlled_torch_fomaml_formal_v1_commit_752fb27/`. It reads the
stored `0/50/100/200` support curves and does not retrain or modify raw
results. Every derived file remains diagnostic with `formal_result=false`.

| Query threshold | Method reached rate | Method mean support steps | Baseline reached rate | Baseline mean support steps | Observation |
|---:|---:|---:|---:|---:|---|
| 0.80 | 1.00 | 0.0 | 1.00 | 33.8 | Method has a floor effect at zero support. |
| 0.90 | 1.00 | 87.6 | 1.00 | 311.0 | Separates methods, but still uses a small number of checkpoints. |
| 0.95 | 1.00 | 172.0 | 1.00 | 349.4 | Best current candidate for a follow-up pilot; still requires validation. |
| 0.975 | 1.00 | 267.4 | 0.80 | 214.5 | Right censoring appears for baseline and ablation-skill. |

At threshold `0.95`, paired support-step savings for method were 177.4 steps
against baseline, 135.4 against ablation-skill, and 143.6 against
ablation-knowledge on average over the five seeds. These are descriptive
diagnostics: support curves contain only four checkpoints, the controlled task
family is small, and the source run is not a formal result.

The next pilot should test `0.95` as a candidate threshold, preserve the
right-censoring rule, and report task-stratified curves. The value must not be
retroactively substituted into prior formal claims.
