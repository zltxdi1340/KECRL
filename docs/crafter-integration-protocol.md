# Crafter Integration Protocol

Status: integration draft; not a frozen formal experiment. Controlled discrete
runs remain archived diagnostics. Crafter 1.8.3 is the installed smoke version.

## Observation Boundary

The learning interface exposes the rendered RGB image (64, 64, 3, uint8),
discrete action IDs, environment reward/termination, and structured inventory.
The inventory extension follows the existing Capability decision. The adapter
filters native info with an explicit allowlist: inventory only. Global semantic
maps, player coordinates, achievements, recipes, and private environment fields
are not returned to learners. An independent evaluator may later use achievement
statistics, under an explicitly separate protocol.

Native reset has no inventory info. Inventory remains unknown until a public
step observation supplies it; reset does not read private player state or take
an implicit noop. Unknown inventory must not be interpreted as empty inventory.

## Candidate Tasks and Evidence

Candidate resource targets include collecting wood/stone and acquiring tools.
The task list, Skill Families, context encoding, Capability schemas, episode
split, rewards, and success criteria remain pending. These candidates do not
freeze Crafter recipes or replace environment mechanism validation.

Runtime evidence may contain before/after public inventory, scope, provenance,
and intervention metadata. Ordinary transitions retain evidence_validity=unknown
and performed=false. Policy failure, death, timeout, or success cannot confirm or
refute structural claims. Interventions, paired worlds, a reference verifier,
validation budgets, and mapping to 1/0/bottom remain to be implemented. Full
maps/rules used by that future verifier must stay outside the learning interface.

## Smoke Scope

The smoke downsamples RGB images to 8x8 on the selected device and uses the
existing categorical REINFORCE implementation for one short rollout/update.
Inventory is exposed by the adapter but is not an input to this diagnostic
policy. No formal task, SPT/SPI/Module, qualification, meta-learning, mechanism
confirmation, or Continual Learning Pipeline integration is claimed.

It records config, source hashes, commit and dirty state, package versions,
PID, visible GPU, runtime, CUDA memory, reward, gradients, parameter change,
and one unknown public transition evidence example. It saves JSON/CSV, a real
rendered frame, full log, dependency snapshot, and GPU captures in a fresh run
directory. All outputs carry formal_result=false.

The candidate task and role preview is recorded in
`configs/crafter_task_protocol_v1.yaml` and
`docs/crafter-task-protocol-v1.md`; it is intentionally not a formal freeze.

A single-transition Pipeline smoke is recorded under
`results/crafter_policy_module_pipeline_smoke/result.json`. It uses a clearly
marked fixture mechanism and fixture qualification record, but the returned
Module ID is checked against and executed by a real categorical policy wrapper.
This verifies `TransitionResult`, Knowledge Evidence, Skill Feedback, task
version routing, and Module-to-policy execution. The observed wood target
remained incomplete, the task result was `continued`, and ordinary evidence
remained `unknown`; this is not a Crafter success, qualification result, or
method comparison. The policy is not updated during this smoke.

Next: freeze task/observation contracts and independent episode roles, connect
real Policy initialization to context-conditioned FOMAML, then connect qualified
Modules to actual policy execution. Baseline retains Module reuse with both
Evolution components off; ablations switch each separately. Seeds 0-4 are the
user-selected candidate set. Crafter budgets, qualification thresholds, SPT
acceptance/non-regression values, and formal statistics remain pending.

The qualification smoke evaluates a candidate policy on fresh episodes before
creating a Module. It records samples, target successes, contract passes, and
exceptions. A failed gate keeps the candidate out of the Skill Library; a pass
at this diagnostic scale would still not freeze formal thresholds or establish
method effectiveness.

The first GPU qualification smoke used 10 episodes for `collect_wood`: all 10
executions passed the contract path, but 0 reached the inventory target within
32 steps. The candidate therefore failed the success gate and no Module was
registered. This is a diagnostic negative result for an untrained policy, not a
method comparison or an environment impossibility claim.

The real paired-world reference runner is implemented in
`src/counterfactual/crafter_reference_runner.py` and configured by
`configs/crafter_paired_reference_v1.yaml`. It uses Crafter's private world
state and transition rules only on the verifier side. For seeds 0--4, the
oracle planner reached the wood-pickaxe target in every baseline world within
11--15 actions (including the public-inventory noop). Removing all tree
materials in the paired intervention world made the target
`PROVEN_UNREACHABLE` in all five worlds. No episode policy was run. This is an
executable structural-verification diagnostic, not a learner result or formal
method comparison; its result records commit, dirty state, source hashes,
runtime version, paired outcomes, and Knowledge Evidence. The recorded run is
`results/crafter_paired_reference_v1_v2/result.json`.

The paired-world output is connected to the Beta-Binomial Knowledge Evolution
diagnostic by `experiments/run_crafter_knowledge_evolution.py`. Using the five
real counterevidence cases gives support `0`, counterevidence `5`, and
effective sample count `5`; with the diagnostic `n_min=10`, the proposition
remains `candidate` and the mechanism remains `testing`. This deliberately
does not reject the mechanism from an under-budget run. The trace is stored at
`results/crafter_knowledge_evolution_v1/result.json` and is not a formal
experiment result.

The result was independently rerun after commit `d5224ef` with only visible
GPU1 (`CUDA_VISIBLE_DEVICES=1`). The fresh trace is stored under
`results/crafter_qualification_smoke_clean_v2/`: CUDA was resolved and a CUDA
policy tensor was verified; 10/10 contract checks passed, 0/10 episodes reached
the wood target, and the candidate remained unregistered. The run took 6.11 s;
the environment check reported Python 3.10.22, PyTorch 2.14.1+cu130, CUDA 13.0,
and an RTX 4090. The repository regression suite remained green (`39 passed`).

A separate horizon diagnostic at the same commit used the untrained policy,
the same five seeds and two episodes per seed, but allowed 256 steps. It
reached wood in 2/10 episodes (0.20), so the longer horizon exposes occasional
random success while remaining far below the candidate 0.80 gate. Its trace is
stored under `results/crafter_horizon256_random_diagnostic/`; this is still a
diagnostic and does not justify changing the formal horizon or qualification
thresholds.

The next backend diagnostic used the real RGB policy path with a short
REINFORCE loop on CUDA (`results/crafter_policy_training_smoke_clean_v2/`). At
the recorded commit `ea1f7ce`, it ran 20 training episodes and 20 optimizer
updates; policy parameters changed and CUDA tensors were verified. On 10 fresh
evaluation episodes the trained smoke policy reached wood in 4 episodes (0.40).
No Module qualification or registration was attempted. This confirms the
policy update and held-out evaluation plumbing, while the small run provides no
evidence of KECRL method effectiveness and is not a formal result.

The commit-aligned support/query adaptation smoke is stored under
`results/crafter_policy_adaptation_smoke_clean_v2/` and was run at commit
`a0d6c3b` on the idle visible GPU0. For each of five seeds, one support
episode was used for a single policy adaptation update and two disjoint query
episodes were evaluated with matched action RNG. CUDA tensors and the disjoint
split were verified; query success was 4/10 before adaptation and 6/10 after
adaptation. No SPT candidate, Knowledge state, or Module was changed. This is
only evidence that the support/query policy plumbing executes; the small
stochastic diagnostic is not a FOMAML result or a formal method comparison.

The commit-aligned CUDA policy FOMAML smoke is stored under
`results/crafter_fomaml_smoke_clean_v2/` and ran at commit `2ace696` on the
idle visible GPU1. It performed five support/query meta updates with disjoint
episode seeds, verified CUDA tensors and changed the active policy parameters.
On ten independent evaluation episodes, success was 1/10 before and 1/10
after the meta updates. No SPT candidate or Module was created. This validates
the first-order policy gradient plumbing only; it is not evidence of method
effectiveness and is not a formal experiment.

Commit `d39d143` adds a context-conditioned policy initializer and the
`crafter_context_policy_smoke` runner. The initializer creates a fresh policy
clone with a context-generated action-bias offset and verifies that the shared
template remains unchanged; it does not update an SPT, register a Module, or
write Knowledge state. The associated CUDA smoke is pending because both
visible GPUs were occupied by external processes at implementation time. No
CUDA result is claimed until a device is available; a future run must use the
commit-aligned config, a fresh result directory, and retain
`formal_result=false`.

The commit-aligned CUDA run completed at commit `fce69a4` in
`results/crafter_context_policy_smoke_clean_v2/`, using only visible GPU0.
CUDA was resolved successfully with PyTorch 2.14.1+cu130 on an RTX 4090, and
CUDA tensors were verified. The shared template stayed unchanged and the two
candidate contexts generated distinct initial action logits. Each context was
evaluated on two independent query seeds at the 64-step diagnostic horizon;
neither reached the wood target. This confirms context-conditioned
initialization and query execution plumbing only. It does not show that the
context helps learning, and it does not perform policy training, FOMAML,
qualification, SPT updates, Module registration, or a formal comparison.

The initial `clean_v1` invocation was rejected before execution because the
outer capture script pre-created the output directory, while the runner
correctly refuses to overwrite an existing directory. Its trace is retained
under `results/crafter_context_policy_smoke_clean_v1_failed_preflight/` and
is not an experiment result.

Commit `d8d10ed` adds a first-order context-conditioned policy FOMAML update.
Its commit-aligned Crafter smoke is stored in
`results/crafter_context_fomaml_smoke_clean_v1/` and ran on visible GPU0.
PyTorch 2.14.1+cu130 resolved CUDA and verified CUDA policy tensors. The
training support/query seeds and independent evaluation support/query seeds
were disjoint. Across the wood and stone gather task contexts, the candidate's
context generator and policy template both changed while the active SPT state
remained unchanged. The candidate was not accepted, no active pointer was
switched, and no Module was registered. On the two tiny independent query
tasks, success was 0/2 before and 0/2 after the candidate update. This run
validates the context-to-generator FOMAML gradient path and candidate version
isolation only; it provides no evidence of improved query learning efficiency
and is not a formal result.

The expanded pilot at commit `7c0fc67` is stored in
`results/crafter_context_fomaml_pilot_v1/` and ran on visible GPU0 with the
same CUDA/PyTorch stack. It used a 256-step horizon and ten task instances
covering wood/stone gather contexts labelled across seed IDs 0--4. Training
support/query and independent evaluation support/query seeds were disjoint;
CUDA tensors were verified, active SPT state remained unchanged, and both the
candidate context generator and policy template changed. The independent
evaluation query result was 0/10 before and 1/10 after the candidate update;
mean query loss was 3.6006 before and 2.6160 after. The one candidate success
was `collect_wood_seed3`. Candidate acceptance was not evaluated, no active
pointer was switched, and no Module was registered. Because these ten task
instances were processed in one shared candidate outer update, this is a
single pilot diagnostic rather than five independent training repetitions; its
success difference cannot support formal seed statistics or an effectiveness
claim.

The independent replica diagnostic at commit `172e320` is stored in
`results/crafter_context_fomaml_independent_v1/`. It ran one serial Python
process on visible GPU0 for seeds 0--4. Each replica used a separate candidate
outer update, disjoint training/evaluation and support/query episode seeds, and
a deterministic seed offset for action sampling. CUDA tensors were verified in
all five replicas; active SPT state stayed unchanged, both candidate parameter
groups changed, and no SPT pointer or Module was registered. Per-seed
independent query success rates before versus after the update were
`0.0/0.0`, `0.3/0.4`, `0.3/0.3`, `0.2/0.2`, and `0.2/0.1`. The across-seed
mean was `0.20` before and `0.20` after, with mean per-seed change `0.00`; mean
query loss changed from `0.4324` to `0.5434` (delta `+0.1111`). This is a
small non-formal regression diagnostic: it shows no consistent query-efficiency
improvement in this Crafter setup and cannot support a method-effectiveness
claim or a frozen hyperparameter decision.

The commit-aligned repeat at `84e604a` is stored under
`results/crafter_context_fomaml_independent_84e604a/` and used one serial
Python process with only visible GPU0. All five replicas resolved `cuda` with
PyTorch 2.14.1+cu130 on an RTX 4090; training/evaluation and support/query
episode seeds were disjoint, CUDA tensors were verified, active SPT state was
unchanged, and both candidate parameter groups changed. Independent query
success rates before versus after the candidate update were `0.0/0.1`,
`0.3/0.4`, `0.3/0.2`, `0.2/0.2`, and `0.0/0.0`. The across-seed mean changed
from `0.16` to `0.18` (mean per-seed change `+0.02`), with about 6.49 seconds
of measured replica time. This repeat remains a non-formal feasibility
diagnostic: the small change does not establish query-efficiency improvement,
candidate acceptance, Module qualification, or a formal comparison.

The repeated-query diagnostic at commit `a05042e` is stored in
`results/crafter_context_fomaml_diverse_query3_v1/`. It keeps one support
adaptation per task and evaluates three independent query episodes per task,
with a documented query-seed stride, across the same six targets and five
independent replicas. All episode roles remained disjoint and all replicas
verified CUDA tensors and candidate isolation. Across 90 active query episodes,
success was 4/90 (`0.0444`); after the candidate update it was 1/90
(`0.0111`). Collect wood accounted for all successes (4/15 active and 1/15
candidate); collect stone, collect coal, and all three pickaxe targets were
0/15 in both phases. Mean normalized query loss changed from `0.1225` to
`0.1009`, but this remains a diagnostic quantity rather than a frozen
cross-task efficiency estimator. Repeating query episodes makes the variance
visible; it does not establish a benefit and suggests that the current policy
and reward setup cannot learn the diverse targets within this one-update
budget.

The diverse-task diagnostic at commit `ace14d8` is stored in
`results/crafter_context_fomaml_diverse_v1/`. It used six distinct task targets
from the candidate Crafter protocol (wood, stone, coal, and three pickaxe
targets), six-dimensional one-hot contexts, and five independent candidate
updates on visible GPU0. All replicas verified CUDA tensors, disjoint episode
roles, unchanged active SPT state, and candidate-only parameter changes. Across
30 independent evaluation episodes, active and candidate query success were
both 1/30 (`0.0333`); the only success was collect wood. By task, collect wood
was 1/5 before and 1/5 after, while the other five targets were 0/5 in both
phases. Mean query loss changed from `-1.2346` to `-1.0209`, which is not used
as a standalone effectiveness claim because the current normalized REINFORCE
loss is not yet a frozen cross-task efficiency estimator. This remains a
diagnostic task-diversity check, not a formal comparison; the craft targets in
particular require a trained multi-step policy and should not be interpreted as
evidence that those tasks are unreachable.

The paired entropy diagnostic at commit `081d1b6` is stored in
`results/crafter_context_fomaml_diverse_query3_entropy_paired_v1/`. It uses the
same six tasks, five replica seeds, training seeds, support seeds, and query
seeds as the non-entropy query3 run, with `entropy_coef=0.01` as the only
backend change. Active-before success stayed 4/90 in both runs. Candidate-after
success was 1/90 without entropy and 3/90 with entropy, with both additional
successes on collect wood; all other task families remained 0/15. Candidate
mean normalized query loss changed from `0.1009` to `0.1284`, so the success
increase is not a consistent loss improvement. This supports retaining entropy
regularization as a hyperparameter-scan candidate, but does not freeze `0.01`
or establish method effectiveness.
