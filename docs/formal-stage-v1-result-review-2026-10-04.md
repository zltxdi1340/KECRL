# Formal Stage v1 Candidate Result Review

The candidate controlled comparison completed 20 runs on GPU1: four variants
and five seeds. Every run used CUDA tensors, passed qualification, reached the
query threshold, and produced JSON/CSV/checkpoint artifacts. The complete
metadata is in `results/formal_stage_v1/run_metadata.json`.

The analyzer reports query success `0.995` for both `method` and
`ablation_skill`, versus `0.991` for `baseline` and `ablation_knowledge`.
The paired method delta against baseline is `+0.004`; the deterministic
diagnostic bootstrap interval is `[-0.002, 0.012]`. All Knowledge-enabled
statuses match the controlled manifest truth, and Pipeline completed rate is
`1.0` with query evidence remaining `unknown` as required by the contract.

These outputs remain `formal_result=false`. The interval crosses zero, the
environment is controlled, and the Policy/FOMAML implementation is still the
stage backend. The values therefore support reproducibility and pipeline
validation, not a final paper performance claim.

The runner audit at commit `bd58d20` adds an explicit limitation: its
`ContextConditionedFOMAML` object is a scalar reference implementation and is
not connected to the CUDA categorical policy. The recorded method and
ablation-skill flags therefore do not constitute a policy-level Skill
Evolution comparison. These archived results remain useful for controlled
pipeline, Knowledge, qualification, and provenance checks, but a valid
policy-level comparison requires a torch FOMAML runner whose candidate update
changes the policy used for query evaluation.
