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

The outer-update curve diagnostic at commit `86b89fe` is stored in
`results/controlled_torch_fomaml_stage_v4/`. It predeclared three candidate
outer updates and evaluated the same paired query roles after each update.
Across five seeds, method query success averaged `0.85`, `0.85`, and `0.80`
after updates 1, 2, and 3; ablation-knowledge averaged `0.75`, `0.75`, and
`0.75`; the zero-update baseline was `0.75`; and zero-update ablation-skill
was `0.80`. The method curve therefore did not improve monotonically, which
is a useful warning against selecting the final update by convenience. This
remains a controlled learning-curve diagnostic with `formal_result=false`,
not a formal budget or effectiveness result.

The resampled multi-episode diagnostic run at commit `4319840` is stored in
`results/controlled_torch_fomaml_v5/`. It uses five seeds and four variants;
each Skill Evolution variant performs five outer updates with two fresh
train support/query pairs per task per update. Fixed evaluation uses five
independent support episodes and ten independent query episodes per task, for
40 query episodes per checkpoint. Train support/query IDs are disjoint, the
evaluation query set is fixed across checkpoints, and all 20 runs verify CUDA
policy tensors.

At the final checkpoint, mean query success was `0.905` for method, `0.885`
for ablation-knowledge, `0.870` for ablation-skill, and `0.855` for baseline.
Mean support interaction steps were `56.8`, `68.2`, `66.2`, and `77.2`,
respectively. The paired method versus ablation-skill delta was `+0.035`
success rate (sample SD `0.0627`) and `-9.4` support steps (sample SD `5.08`);
the per-seed success deltas were `[+0.025, +0.050, -0.050, +0.125, +0.025]`.
The method curve was not monotonic in the earlier three-update diagnostic,
but v5 gives a more stable multi-episode estimate. These values remain
diagnostic: the policy is still a small controlled backend, qualification and
SPT acceptance are not part of this runner, and `formal_result=false`.

The v5 analysis artifacts are in
`results/controlled_torch_fomaml_v5/analysis/`. The analyzer validates the
historical run against `datasets/discrete_resource_manifest_stage_v1.json`
because these raw results predate the recorded `role_episode_ids_disjoint`
field. It reports the full outer curve, a predeclared final-checkpoint rule,
per-seed paired deltas, and thresholded query-learning efficiency with right
censoring. At threshold `0.8`, method reached the threshold for all five
seeds, ablation-knowledge for all five, and baseline and ablation-skill for
four of five seeds. The method versus baseline paired success delta was
`+0.050` (bootstrap interval `[0.000, 0.100]`), and the method versus
ablation-knowledge delta was `+0.020` (interval `[-0.010, 0.050]`). The
method versus ablation-skill success interval crosses zero, so these results
do not establish a formal effectiveness claim. Bootstrap intervals are
descriptive only; all generated analysis files retain `formal_result=false`.

The next candidate runner, `experiments/train_controlled_torch_fomaml_formal_v1.py`,
adds the previously missing held-out qualification and SPT validation paths.
Its configuration is `configs/controlled_torch_fomaml_formal_v1.yaml`, and the
20-run GPU0 execution is archived under
`results/controlled_torch_fomaml_formal_v1/`. The run used the same five seeds,
four variants, disjoint role manifest, CUDA policy tensors, five Skill outer
updates, 20 qualification episodes per task, and two independent SPT validation
batches of 10 episodes per task. It writes per-run qualification and SPT
decision records plus command, environment, dependency, Git, and GPU
provenance files.

This candidate run is not promoted to a formal result. Qualification passed in
4/5 seeds for every variant (`0.80` pass rate); seed 0 failed the success
threshold in every variant while contract pass rates remained `1.0`. SPT
candidate decisions for `method` were accepted in 2/5 seeds, rejected in 3/5;
the rejections include insufficient improvement or an existing-SPI regression.
The final query-success means were `0.932` (method), `0.917` (baseline),
`0.930` (ablation-knowledge), and `0.928` (ablation-skill), but they are
descriptive only because qualification and SPT decisions are heterogeneous and
the controlled policy/task budget remains small. All outputs keep
`formal_result=false`; the next review must decide whether to enlarge the
qualification/validation budgets or revise the candidate protocol before any
paper claim.

After the design freeze gates were implemented at commit `6319c8c`, the
candidate runner was rerun on GPU0 in
`results/controlled_torch_fomaml_formal_v1_commit_6319c8c/`. All 20 runs record
that commit, verify CUDA tensors, retain `formal_result=false`, and include the
configuration, command, environment, dependency, Git, and GPU provenance.
The runner now reports four support checkpoints (`0, 50, 100, 200`), evaluates
qualification separately for each task/SPI, and registers a Module only when
all task gates pass. Qualification/module registration passed for 3/5 method
seeds, 1/5 baseline seeds, 3/5 ablation-knowledge seeds, and 1/5 ablation-skill
seeds. Method SPT decisions were accepted for seeds 0 and 1 and rejected for
seeds 2--4; baseline and ablation-skill keep the active SPT because Skill
Evolution is disabled. These heterogeneous gates are evidence that the
protocol is exercising the intended boundaries, not evidence of final method
effectiveness. The result directory remains diagnostic and is not promoted to
a paper result.

At commit `3fcd533`, the candidate runner also wires the qualified Skill
Library into the controlled execution path and checks Module output targets
during compatibility matching. A two-variant seed-0 gate check reused 150 of
200 query episodes through target-compatible Modules and returned
`unavailable` for all 50 `craft_shelter` episodes because that task failed its
held-out qualification gate; all 150 actual Module executions completed. The
compatibility fix prevents a Module for one resource target from being reused
for a different target. This is an interface and execution-path validation,
not a performance result.

The complete rerun at HEAD `752fb27` is archived in
`results/controlled_torch_fomaml_formal_v1_commit_752fb27/`. All 20 runs carry
this commit and the new `module_reuse.json` records. Mean query success stayed
at `0.932` for method, `0.917` for baseline, `0.930` for ablation-knowledge,
and `0.928` for ablation-skill. Mean Module reuse rates were `0.90`, `0.80`,
`0.90`, and `0.80`, respectively; unavailable query transitions corresponded
to tasks without a qualified Module and were not counted as execution
failures. These are still controlled diagnostic measurements with
`formal_result=false`.
