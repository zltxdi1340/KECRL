# Task-Stratified Query Efficiency Pilot

The pilot at
`results/controlled_torch_fomaml_formal_v1_pilot095_70c25cd/` reran the
controlled candidate protocol with query threshold `0.95` and added per-task
query curves at support checkpoints `0, 50, 100, 200`. It used one visible GPU,
five seeds, four variants, and commit `70c25cd`. The output remains
`formal_result=false`.

At support budget zero, the method task-level query success means were:

| Task | Method | Baseline | Ablation-knowledge | Ablation-skill |
|---|---:|---:|---:|---:|
| `gather_wood` | 1.00 | 0.96 | 0.94 | 1.00 |
| `craft_tool` | 0.82 | 0.76 | 0.82 | 0.76 |
| `craft_shelter` | 0.84 | 0.78 | 0.84 | 0.78 |
| `use_tool` | 0.88 | 0.76 | 0.88 | 0.76 |

The aggregate method curve reaches `0.95` after a mean of 172 support
interaction steps; baseline requires 349.4, ablation-knowledge 315.6, and
ablation-skill 307.4. These are descriptive pilot values. The task table shows
that the aggregate threshold can hide task-level differences, especially for
the craft tasks, so the next formal protocol should report task-stratified
curves alongside the aggregate endpoint. No threshold has been promoted to a
final paper metric.
