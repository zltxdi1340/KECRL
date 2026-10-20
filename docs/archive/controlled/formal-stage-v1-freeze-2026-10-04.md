# Formal Stage v1 Candidate Freeze

This file defines the executable candidate configuration for the first formal
controlled comparison. It remains a candidate until the run is completed and
reviewed; all generated outputs must retain `formal_result=false` during the
run.

- Environment: controlled discrete resource environment, manifest
  `datasets/discrete_resource_manifest_stage_v1.json`.
- Seeds: `0,1,2,3,4`; five disjoint role sets per seed.
- Variants: `method`, `baseline`, `ablation_knowledge`, `ablation_skill`.
- Policy backend: CUDA PyTorch categorical REINFORCE; FOMAML path remains the
  current reference implementation.
- Knowledge action prior: strength `1.5`, enabled only when Knowledge
  Evolution is enabled (`method` and `ablation_skill`). Baseline and
  `ablation_knowledge` use strength `0.0` through the component switch.
- Qualification: 20 samples, success rate `>=0.8`, contract pass rate `1.0`.
- Primary metric: independent query learning efficiency with support curve
  checkpoints `0,50,100,200` and query threshold `0.8`.
- Statistical analysis: `configs/statistical_analysis_v1.yaml`; paired by seed
  against baseline, reporting per-seed deltas, means, standard deviations, and
  diagnostic bootstrap intervals.

This freeze does not promote prior smoke, holdout, or controlled-stage outputs
to paper results. Crafter remains outside this stage.
