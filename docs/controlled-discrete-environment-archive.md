# Controlled Discrete Environment Archive

Status: archived as an interface and reproducibility benchmark. It is not the
target environment for the method-effectiveness claim.

The environment is implemented in `src/environments/discrete_resources.py` and
uses deterministic manifests under `datasets/`. It contains gather, craft, and
tool-use resource tasks with disjoint train, support, query, qualification, and
SPT-validation roles. The GPU runs are retained as regression evidence with
`formal_result=false`.

## Why it did not demonstrate effectiveness

The method and baseline both reach near-saturated query success, leaving too
little headroom for Knowledge Evolution or Skill Evolution to improve the
primary metric. The task rules are small, deterministic, and directly exposed
through resource prerequisites, so the policy can solve most episodes without
needing persistent mechanism discovery. The controlled mechanism evidence is
also supplied from a fixed manifest, rather than being discovered under noisy
or changing observations. Module qualification and pipeline execution are
therefore tested as contracts, but they do not create a difficult decision
problem. Finally, the current policy backend and short horizons make the
support/query curves saturate quickly; increasing the budget from 200 to 400
episodes did not create a stable method advantage.

These observations are a limitation of the benchmark's discriminative power,
not evidence that the theoretical method is invalid. The environment remains
useful for unit tests, CUDA execution checks, data-contract regression, and
reproducibility checks.

## Archived evidence

- `results/formal_stage_v1`: 20-run controlled candidate comparison.
- `results/formal_stage_v1_regression`: independent GPU reproduction.
- `results/formal_stage_v1_budget400`: budget sensitivity diagnostic.
- `results/holdout_prior_v3` and `results/holdout_prior_v4`: independent-seed
  prior sensitivity and reproduction.
- `results/*/run_metadata.json`, environment captures, and analysis summaries
  preserve configuration, commit, device, and formal-result status.

All outputs remain explicitly non-formal. Large checkpoints and transient logs
are retained on the server but are excluded from the GitHub source archive.
