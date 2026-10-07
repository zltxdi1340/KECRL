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

Task-local composition may also declare `crafter_world_object_setup` targets
for table and furnace placement. The adapter accepts this field only when an
explicit environment wrapper supplies it in the reviewed public info
allowlist. Native Crafter 1.8.3 does not supply such a field, so the adapter
may confirm setup only from the public transition contract: the executed
`place_table`/`place_furnace` action must reduce the corresponding public
inventory by exactly its versioned resource cost. Semantic maps, player
coordinates, achievements, and RGB inference are not used as setup
confirmation. A confirmed setup produces the corresponding public capability
and can be carried to the next step in the same task-local plan; any missing or
ambiguous transition remains `unknown`.

`CrafterTaskPlanExecutor` enforces this plan boundary: the first qualified
Module resets the native environment, while later prerequisite and target
Modules reuse the current observation. A plan reset is explicit via
`begin_episode`; it does not create a persistent world or store a trajectory.

The runtime-only diagnostic in
`experiments/run_crafter_task_local_runtime_diagnostic.py` exercises the
`collect_stone` chain with fixture-qualified Modules and the real RGB Policy
executor. The two-seed run reached only the first `gather_wood_3` step and
returned `continued` after the 32-step limit, with one unknown ordinary
Knowledge evidence record per episode. This confirms routing and boundary
behavior; it is not a qualification, learnability, or method result.

The task-local composition candidate now decomposes the former `wood >= 3`
prerequisite into three repeated `wood >= 1` transitions in the same episode.
This preserves the formal primary targets and lets a qualified low-threshold
gather Module be reused for each unit while public inventory accumulates. The
decomposition is a candidate protocol change and still requires independent
qualification and end-to-end validation.

The non-formal diagnostic
`experiments/run_crafter_task_local_oracle_pipeline_diagnostic.py` now runs the
`obtain_wood_pickaxe` plan through this complete boundary. A verifier-side
public-action-consistent script is replayed on a fresh learner-facing adapter;
fixture Modules are requested through the Skill Library, routed by
`CrafterTaskPlanExecutor`, and executed through the Continual Learning
Pipeline. Five seeds completed all five plan steps, with the same gather Module
reused three times, table setup confirmed from the public inventory decrement,
and five `unknown` ordinary Knowledge Evidence records per episode. This is a
state-transfer and routing diagnostic only. It does not train, qualify, or
register a real Module and remains `formal_result=false`.

The same runner also has a separate `collect_stone` configuration. Its
verifier script first reaches the wood-pickaxe prerequisite, then gathers a
stone through the private-map oracle; replay still uses only public adapter
actions. Five seeds completed the six-step candidate chain, and the
wood-pickaxe fact was available to the final stone request. This is still a
fixture and routing diagnostic, not a real-policy qualification or formal
result.

The dependency-aware candidate now has a separate v2 manifest with explicit
`setup_table` and `setup_furnace` task boundaries. This removes the audit's
missing-producer ambiguity for world objects while keeping adapter support and
persistent execution pending. The manifest is a protocol candidate only and
does not authorize formal training.

The persistent candidate order has a separate boundary diagnostic using
`CrafterContinualSession`. It confirms that public inventory and table/furnace
setup survive task boundaries on the successful replay. After the verifier
started checking all reachable material candidates, all five seeds yielded a
complete route through four stone and furnace. All five routes replayed through
the learner-facing adapter after furnace facing was changed to public movement
actions. The verifier now checks all reachable material candidates instead of
declaring failure after an inaccessible nearest resource; the current replay
diagnostic reaches `5/5` verifier routes and completes `5/5` replays, with no
first public trace divergence. This remains a boundary diagnostic and does not
qualify a learned Module.

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

The real-policy qualification and Pipeline routing diagnostic at commit
`afa24b0` is stored in `results/crafter_module_pipeline_diagnostic_afa24b0/`.
It used the CUDA RGB policy wrapper, five seeds, ten held-out qualification
episodes per task, and the fixture mechanism required by this diagnostic. All
60 qualification episodes completed the contract path (`contract_rate=1.0`),
but `collect_wood` reached the target only 3/10 times and the other five task
targets reached 0/10. No candidate passed the `0.8` success gate, so no Module
was registered and all 60 Pipeline requests correctly returned `unavailable`
without calling the executor or producing feedback. This confirms the
unavailable routing and hard qualification behavior; it also leaves the formal
gate `qualified_module_pipeline_connected` blocked until a trained Policy is
qualified on independent data. The run remains `formal_result=false`.

The split-aware feasibility pilot at commit `afa24b0` is stored in
`results/crafter_split_feasibility_pilot_afa24b0/` and was run as one serial
process on visible GPU0 with the 256-step candidate horizon. It used
`collect_wood` and `collect_stone` for the training-side update and all six
candidate tasks for independent evaluation, with three query repeats per task.
All five replicas resolved CUDA, verified CUDA policy tensors, kept training,
support, and query roles disjoint, changed only the candidate policy, and left
the active policy/SPT and Module/Knowledge state untouched. Mean independent
query success was `0.0333` before and `0.0556` after the candidate update; all
observed successes were on `collect_wood`, while `collect_stone`, `collect_coal`,
and the three pickaxe targets remained at zero in every replica. This is a
feasibility and cost diagnostic showing that the current budget does not make
the diverse tasks reliably learnable. It is not a formal comparison, threshold
validation, Module qualification, or method-effectiveness result; the aggregate
and per-replica files retain `formal_result=false`.

The first end-to-end real-policy connection diagnostic at commit `79c0507` is
stored in `results/crafter_fomaml_module_pipeline_diagnostic_79c0507/` and is
configured by `configs/crafter_fomaml_module_pipeline_diagnostic_v1.yaml`. It
connects one real RGB gather task through context-conditioned FOMAML, an
independent adaptation support episode, held-out qualification, Module
registration/reuse, and the actual Pipeline. All five replicas verified CUDA,
kept role seeds disjoint, left the active SPT unchanged, and changed the
candidate Policy. Every replica registered and reused a Module under the
explicit diagnostic qualification threshold `0.1`; qualification success was
`0.1--0.4`, and fresh Pipeline episodes returned only `completed` or
`continued`. Each reused execution generated one ordinary Knowledge Evidence
record with `evidence_validity=unknown` and one Skill Feedback record. The
threshold is deliberately below the formal candidate `0.8` and is plumbing
diagnostic metadata only, so this run does not validate formal qualification,
SPT acceptance, or method effectiveness. The aggregate remains
`formal_result=false`.

At commit `b58c0e4`, the paired verifier was rerun in
`results/crafter_paired_reference_b58c0e4/` and consumed by Knowledge Evolution
under `results/crafter_knowledge_evolution_b58c0e4/`. All five baseline worlds
were `FOUND`, all five intervention worlds were
`PROVEN_UNREACHABLE`, and every emitted evidence record had the expected
`knowledge_observation=0` and `evidence_validity=invalid`. The read-only
boundary audit is stored in `results/crafter_knowledge_boundary_b58c0e4/` and
passed scope matching, pair/world uniqueness, three-state mapping, forbidden
skill-field checks, duplicate-safe count checks, and commit provenance. The
connected Knowledge result remains `support=0`, `counterevidence=5`,
`effective_n=5`, `n_min=10`, status `candidate`; it therefore does not reject
or confirm a mechanism and does not permit formal training. UNKNOWN mapping is
covered by the verifier and boundary-audit tests but was not produced by these
five complete reference searches.

The read-only action/inventory diagnostic at
`results/crafter_action_inventory_diagnostic_worktree/result.json` confirms the
adapter contract: Crafter exposes 17 actions in the native order and every
action step returns public inventory. Two deterministic action-cycle rollouts
ended by health depletion with zero wood, which is expected for a non-policy
probe. No reset, action mapping, or inventory allowlist fault was observed.

The same end-to-end real-policy connection diagnostic was rerun at commit
`1a889b7` under `results/crafter_fomaml_module_pipeline_diagnostic_1a889b7/`.
All five replicas again verified CUDA, preserved the active SPT, changed the
candidate Policy, registered and reused a Module, and routed execution through
the actual Pipeline. Qualification rates for `collect_wood` were `0.3`, `0.4`,
`0.1`, `0.2`, and `0.1`; Pipeline outcomes remained within `completed` and
`continued`. The run therefore confirms current-code plumbing continuity, but
the diagnostic threshold is still `0.1` rather than the candidate formal `0.8`.
It does not validate formal qualification or permit formal training.
