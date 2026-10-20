# Controlled Stage Comparison Freeze

> This file freezes the first executable comparison configuration. It does not
> turn controlled runs, smoke runs, or reference FOMAML results into paper
> results. Every generated result remains `formal_result=false` until the
> remaining backend and statistical review gates are completed.

## Scope

- Environment: deterministic controlled discrete resource environment.
- Task family: mechanism identification and skill meta-learning over discrete
  resource tasks (`gather_wood`, `craft_tool`, `craft_shelter`, `use_tool`).
- Crafter is deferred to a later environment-validation stage.
- Policy backend: CUDA PyTorch categorical REINFORCE used by the current stage
  runner; the context-conditioned FOMAML path remains a reference implementation.
- GPU allocation: one GPU only, with `CUDA_VISIBLE_DEVICES=1`.

## Comparisons

- `method`: Knowledge Evolution, Skill Evolution, and compatible Module reuse.
- `baseline`: Module reuse retained; Knowledge Evolution and Skill Evolution off.
- `ablation_knowledge`: Knowledge Evolution off; Skill Evolution and Module reuse on.
- `ablation_skill`: Skill Evolution off; Knowledge Evolution and Module reuse on.

All variants use the same manifest, episode roles, seeds, qualification rules,
and evaluation budget. Only the named evolution component changes.

## Frozen run values

- Seeds: `0,1,2,3,4`.
- Episodes per role and seed: `200` for train, support, query,
  qualification, and SPT validation.
- Qualification: at least `20` samples, success rate `>=0.8`, contract pass
  rate `1.0`; hard contract violations are counted separately.
- Knowledge candidate values: `n_min=10`, `tau_confirm=0.8`,
  `tau_reject=0.2`, confidence `0.95`, budget `200` per seed.
- SPT candidate rule: query-learning-efficiency improvement `>=10%`, existing
  SPI regression `<=0.05`, and two independent validation batches. These are
  candidate values and must be checked against the resulting data before any
  claim of final statistical adequacy.
- Primary metric: independent query learning efficiency. Secondary metrics are
  recorded in `configs/controlled_stage_v1.yaml`.

## Reproducibility

The executable configuration is `configs/controlled_stage_v1.yaml`; the
manifest is `datasets/discrete_resource_manifest_stage_v1.json`. Each run must
record the config, manifest, seed, Git commit, Python/PyTorch/CUDA/GPU details,
logs, JSON/CSV output, checkpoint, and failures in a new result directory.

## Release gate

The run is a controlled stage comparison and not a final paper experiment.
Before promotion to formal results, review the real policy/backend scope,
independent query-efficiency estimator, cross-seed aggregation, statistical
procedure, and environment validation. GPU smoke, unit tests, and this stage
comparison cannot satisfy that gate by themselves.
