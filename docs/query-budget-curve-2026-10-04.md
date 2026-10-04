# Support Budget Curve Diagnostic

Holdout v3 evaluates the independent query set at support checkpoints
`0,50,100,200` episodes. Query episodes are read-only at each checkpoint and
are never used for policy updates. All outputs remain `formal_result=false`.

Mean query success curves over seeds `5..9`:

| Prior | 0 support | 50 support | 100 support | 200 support |
|---:|---:|---:|---:|---:|
| 0.0, all variants | 0.971 | 0.976 | 0.986 | 0.993 |
| 1.5, Knowledge-enabled | 0.954 | 0.974 | 0.964 | 0.991 |
| 1.5, baseline/ablation-K | 0.971 | 0.976 | 0.986 | 0.993 |

The curve confirms that the prior is only active for Knowledge-enabled
variants. The strength `1.5` curve has early fluctuations and does not
dominate the no-prior curve at every budget. It remains a candidate diagnostic
setting, not a final hyperparameter or a formal efficiency claim.

Raw runs are under `results/holdout_prior_v3/strength_*`.
