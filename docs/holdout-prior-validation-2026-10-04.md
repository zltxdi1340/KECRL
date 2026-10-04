# Holdout Validation of Knowledge Action Prior

This diagnostic uses independent seeds `5,6,7,8,9`, which were not used in
the prior-strength scan. The task family, role budgets, qualification rules,
and component switches are unchanged. Both runs use GPU1 and remain
`formal_result=false`.

| Prior strength | Method query success | Ablation-S query success | Baseline query success | Qualification |
|---:|---:|---:|---:|---:|
| 0.0 | 0.989 | 0.971 | 0.971 | 1.00 |
| 1.5 | 0.991 | 0.954 | 0.971 | 1.00 |

Knowledge status accuracy was `1.0` for both Knowledge-enabled variants at
both strengths. On the independent seeds, strength `1.5` slightly improved
the full method but reduced the Skill Evolution ablation's policy success.
This interaction means the prior is currently a method-specific candidate;
it is not a general improvement claim and is not yet a formally frozen
hyperparameter. A formal comparison must define whether the prior is part of
the method only, and must report the corresponding ablation behavior.

Raw outputs are under `results/holdout_prior_v1/strength_*` and indexed by
`results/holdout_prior_v1/scan_index.json`.
