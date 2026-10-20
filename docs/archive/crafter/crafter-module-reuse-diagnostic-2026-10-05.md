# Crafter Qualified Module Reuse Diagnostic (2026-10-05)

This diagnostic exercises the real CUDA policy execution path after an
explicitly marked fixture qualification. The fixture qualification is used to
reach the `reused_module` branch; it is not a policy result, a formal Module,
or evidence of method effectiveness.

## Provenance

- Runner: `experiments/run_crafter_module_reuse_diagnostic.py`
- Config: `configs/crafter_module_reuse_diagnostic_v1.yaml`
- Commit: `3174b5a080804257b92b347136da83a8ab2d11dc`
- Device: CUDA on one RTX 4090
- Task: `collect_wood`
- Episode horizon: 8 steps
- Formal-result flag: `false`

## Observed contract path

1. A fixture `SPI` and implementation contract were created for the public
   `inventory_at_least(wood, 1)` target.
2. The fixture-qualified Module was requested through the Skill Library.
3. The response status was `reused_module` and the returned Module ID matched
   the qualified record.
4. The real CUDA policy executed through `CrafterPolicyModuleExecutor`.
5. The transition returned `target_achieved=false` and execution status
   `terminated`; the task-level result was `continued` under the reviewed
   task-result mapping.
6. The Pipeline recorded exactly one ordinary Knowledge Evidence item with
   `evidence_validity=unknown` and exactly one Skill Feedback item.

No policy parameters, gradients, trajectory, Module performance, or skill
success rate entered Knowledge Evidence. Knowledge Evolution did not update,
the SPT pointer did not switch, and no formal Module claim was made.

This confirms the qualified reuse and dual-feedback routing boundary. It does
not validate real policy qualification, Crafter learning efficiency, Knowledge
mechanism discovery, or formal method effectiveness.
