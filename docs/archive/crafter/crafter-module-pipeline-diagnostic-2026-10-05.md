# Crafter Module and Pipeline Diagnostic (2026-10-05)

This diagnostic checks the qualification gate and the unavailable routing of a
real Crafter policy through the KECRL Skill Library/Pipeline boundary. It is
not a formal result and uses a confirmed fixture mechanism only to exercise
the read path; the fixture is not Knowledge Evolution evidence.

## Provenance

- Runner: `experiments/run_crafter_module_pipeline_diagnostic.py`
- Config: `configs/crafter_module_pipeline_diagnostic_v1.yaml`
- Commit: `90c9af72121a0c14725f9435c76e7390871e28c8`
- Device: CUDA on one RTX 4090
- Seeds: `0..4`
- Six Crafter tasks, 10 qualification episodes and 10 query episodes per task
- Formal-result flag: `false`

## Results

For every task:

- qualification samples: `10`;
- target successes: `0`;
- contract passes: `10/10`;
- success rate: `0.0`;
- Module registered: `false`;
- Pipeline query results: `10 unavailable`, `0 completed`, `0 continued`,
  `0 unknown`.

Across all tasks, no unqualified response reached the executor. The Pipeline
therefore returned `unavailable` as required. Knowledge Evidence and Skill
Feedback counts were both zero because execution was correctly blocked before
the transition stage.

## Interpretation

This run verifies that a real CUDA policy cannot enter the Skill Library merely
because its execution contract passes: the success gate also has to pass. It
also verifies that unavailable implementation requests do not create ordinary
execution evidence or skill feedback. The result is a negative qualification
diagnostic for the current untrained policy, not an environment impossibility
claim and not evidence against KECRL.

The `reused_module` execution branch still needs a separate test with a
policy that actually satisfies the qualification threshold. The existing
fixture Pipeline smoke covers that route structurally; it must not be promoted
to a Crafter performance result.
