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

The new policy-connected runner was smoke-tested at commit `6252e2b` in
`results/controlled_torch_fomaml_gpu_smoke/` on visible GPU0. All four
variants resolved CUDA and verified CUDA policy tensors. `method` and
`ablation_knowledge` changed the candidate Torch policy through the FOMAML
outer update; `baseline` and `ablation_skill` left the policy unchanged by the
outer Skill update. This confirms the missing policy connection has been
implemented in the new diagnostic path. The smoke manifest has no mechanism
evidence field, so Knowledge guidance was recorded as
`knowledge_evidence_unavailable`; consequently this single-seed run does not
provide a valid Knowledge ablation or any formal comparison.

The evidence-backed, paired-role diagnostic at commit `1bed153` is stored in
`results/controlled_torch_fomaml_stage_v3/`. It ran five seeds and all four
variants on visible GPU0 with the real Torch policy FOMAML backend. Support
episodes use the support role and query episodes use the query role; active and
candidate evaluations reuse the same action-sampling seed per task. CUDA
policy tensors were verified in all 20 runs. Knowledge-enabled variants
recorded 60 evidence items and the expected confirmed/rejected/candidate
statuses; the Knowledge-disabled variant recorded no Knowledge evidence.
Method query success changed from 16/20 (`0.80`) to 17/20 (`0.85`), a paired
delta of `+1/20`; baseline stayed 15/20, ablation-knowledge stayed 15/20, and
ablation-skill stayed 16/20. This is the first diagnostic with a policy-level,
paired variant path, but it remains a short controlled run with one outer
update and `formal_result=false`; it is not a formal effectiveness result.
