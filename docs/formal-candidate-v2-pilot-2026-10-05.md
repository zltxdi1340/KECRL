# Controlled Candidate v2 Pilot (2026-10-05)

This document records an expanded-budget candidate run on the controlled
discrete-resource environment. It is an implementation and metric diagnostic;
it is not a promoted paper result.

## Scope and provenance

- Environment: `controlled_discrete_resource_v1`
- Task family: `gather_wood`, `craft_tool`, `craft_shelter`, `use_tool`
- Variants: method, baseline, ablation-knowledge, ablation-skill
- Seeds: `0..4`, 20 variant-seed runs total
- Device: one visible NVIDIA RTX 4090 (`CUDA_VISIBLE_DEVICES=0`)
- Policy backend: `torch_categorical_policy`
- Query threshold: `0.95`
- Support checkpoints: `0, 50, 100, 200`
- Qualification: 30 held-out episodes per task, success threshold `0.8`,
  contract threshold `1.0`
- SPT validation: two batches, 15 episodes per task per batch, minimum
  efficiency improvement `0.10`, maximum existing-SPI regression `0.05`;
  every batch must pass
- Git commit: `1abc9b625a481580270f0dadce64a7bd34c06c01`
- Formal-result flag: `false` for every run and every analysis artifact

The exact command, configuration, environment, dependencies, GPU report, raw
run outputs, and analysis outputs are under
`results/controlled_torch_fomaml_formal_v2_pilot095_1abc9b6/`.

## Aggregate diagnostic values

| variant | query success mean | 0.95 threshold steps | qualification rate | Module reuse | SPT accepted |
|---|---:|---:|---:|---:|---:|
| method | 0.932 | 172.0 | 0.60 | 0.90 | 0.20 |
| baseline | 0.917 | 349.4 | 0.20 | 0.80 | 0.00 |
| ablation-knowledge | 0.930 | 315.6 | 0.60 | 0.90 | 0.60 |
| ablation-skill | 0.928 | 307.4 | 0.20 | 0.80 | 0.00 |

All four variants reached the 0.95 threshold in all five seeds in this pilot.
The method-versus-baseline paired difference in threshold steps is descriptive
only; this run does not freeze a statistical claim or promote a formal result.

## Interpretation and limits

The higher threshold removes much of the 0.8 success-threshold floor effect and
produces a more discriminating support-efficiency diagnostic. The method also
shows the expected higher Module registration/reuse rate in this controlled
implementation because Knowledge and Skill Evolution are enabled. However,
the environment is still a small controlled discrete resource benchmark, the
policy and transition backend are not a Crafter backend, and the sample is only
five seeds. These values therefore support implementation tracing and metric
design, not a claim that KECRL is effective in a general environment.

The SPT gate now records `passes` per validation batch and accepts a candidate
only when every required batch passes contract, improvement, and non-regression
thresholds. Existing Modules remain versioned objects; the candidate does not
rewrite the active SPT or existing Module records.

## Next decision

Keep this pilot archived as a diagnostic. Before any paper-result promotion,
replace the controlled transition adapter with the intended external-environment
adapter, freeze the final baseline and ablations, and rerun the same provenance
and statistical checks with the required independent evaluation budget.
