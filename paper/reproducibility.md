# Reproducibility Statement

All controlled-stage artifacts are generated from versioned JSON
configurations and manifests. The candidate run uses
`configs/formal_stage_v1.yaml`, `configs/statistical_analysis_v1.yaml`, and
`datasets/discrete_resource_manifest_stage_v1.json`. It records five seeds,
disjoint train/support/query/qualification/SPT-validation roles, Git commit,
Python executable, PyTorch/CUDA versions, visible GPU, structured JSON/CSV
results, and Policy checkpoints.

The reproducible command is:

```bash
CUDA_VISIBLE_DEVICES=1 /home/zl202621/.conda/envs/kecrl/bin/python \
  -m experiments.train_controlled --config configs/formal_stage_v1.yaml
```

Analysis is reproduced with:

```bash
/home/zl202621/.conda/envs/kecrl/bin/python \
  -m experiments.analyze_stage_training \
  --root results/formal_stage_v1 \
  --output results/formal_stage_v1/analysis \
  --manifest datasets/discrete_resource_manifest_stage_v1.json
```

The generated tables are exported with
`experiments/export_stage_tables.py`. These artifacts currently describe a
candidate controlled-stage run and retain `formal_result=false`; they are not
a claim that the final paper experiment has been completed.
