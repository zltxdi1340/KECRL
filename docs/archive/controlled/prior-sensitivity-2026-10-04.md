# Knowledge Action Prior Sensitivity

This is a controlled diagnostic scan, not a formal paper result. It used
GPU1, the stage-v1 manifest, seeds `0..4`, and 20 runs per prior strength
(four variants per seed). Every output has `formal_result=false`.

| Prior strength | Method query success | Ablation-S query success | Ablation-S qualification | Baseline query success |
|---:|---:|---:|---:|---:|
| 0.0 | 0.981 | 0.946 | 1.00 | 0.946 |
| 0.5 | 0.984 | 0.959 | 1.00 | 0.946 |
| 1.5 | 0.986 | 0.977 | 1.00 | 0.946 |
| 3.0 | 0.995 | 0.926 | 0.80 | 0.946 |

The method's query success increased with the prior in this controlled scan,
but the Skill Evolution ablation degraded at strength `3.0`, including a drop
in qualification rate. This is evidence of sensitivity and possible
over-conditioning, not a claim of generalization. Strength `1.5` remains the
current candidate because it improved method and ablation-S query success
without reducing qualification in this scan. The value must be rechecked on
an independently frozen task set before formal use.

Raw results are under `results/prior_scan/strength_*`; the scan index is
`results/prior_scan/scan_index.json`.
