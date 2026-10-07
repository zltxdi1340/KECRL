# Crafter Formal Experiment Freeze Candidate (2026-10-05)

This is a machine-checked freeze **draft** for the next Crafter feasibility
work. It is not a formal experiment freeze and it does not authorize formal
training. Every numerical value in the candidate configuration is marked as
`candidate_values_to_validate` unless it is an interface or provenance
invariant.

The machine-readable source is
`configs/crafter_formal_freeze_candidate_v1.yaml`. The read-only audit is
`experiments/audit_crafter_formal_freeze.py`.

## Environment and tasks

The target is Crafter `1.8.3` through the `rgb64_inventory_v1` adapter: RGB
observations of shape `64x64x3` and dtype `uint8`, 17 discrete actions, and an
allowlisted public inventory. Each episode starts a fresh native environment.
The learner does not receive semantic maps, player coordinates, recipes, or
other private fields.

The candidate task split is:

- meta-train: `collect_wood`, `collect_stone`;
- meta-validation: `collect_coal`;
- transfer evaluation: `obtain_wood_pickaxe`, `obtain_stone_pickaxe`,
  `obtain_iron_pickaxe`;
- continual order: all six tasks in the order listed above.

This split is a candidate because the existing implementation has not yet
shown that the craft tasks are learnable under the proposed horizon and policy
budget. A feasibility pilot must test this before a formal freeze.

## Comparisons and metrics

The four candidate variants are:

- method: Knowledge Evolution, Skill Evolution, and compatible Module reuse;
- baseline: Module reuse with both Evolution components disabled;
- ablation-knowledge: Skill Evolution and Module reuse, Knowledge Evolution off;
- ablation-skill: Knowledge Evolution and Module reuse, Skill Evolution off.

The primary candidate metric is independent query learning efficiency: support
interaction steps required to reach a held-out query success threshold, with
unreached thresholds right-censored. Candidate threshold is `0.8`; support
checkpoints are `0, 50, 100, 200, 400`. Results must be reported per seed and
paired against the baseline. Query success/loss, qualification, Module reuse,
task-result categories, Knowledge status accuracy, runtime, and peak GPU
memory are secondary metrics.

## Candidate budget and gates

The draft uses five seeds (`0..4`), 20 train/support/query/qualification
episodes per task, two SPT validation batches of 10 episodes per task, 30
independent evaluation query episodes per task, and a 256-step episode horizon.
The candidate reward records native Crafter reward, adds a success bonus of
`1.0`, and has no step cost. Success uses the public
`inventory_at_least` predicate; success terminates the episode, while timeout
is skill feedback failure and remains UNKNOWN for Knowledge evidence.

Qualification is per task/SPI with 20 samples, success at least `0.8`, and
contract pass rate `1.0`. SPT candidates require every validation batch to pass
at least 10% efficiency improvement, no more than `0.05` existing-SPI success
regression, and no hard contract violation. Failure rejects and keeps the
active SPT; insufficient evidence is inconclusive and keeps the active SPT.

Knowledge validation remains separate from ordinary execution feedback. The
candidate verifier protocol returns `FOUND`, `PROVEN_UNREACHABLE`, or
`UNKNOWN`, uses a per-seed budget of 200 units, and treats ordinary execution
evidence as `unknown`. The reference verifier is not implemented yet.

## Current readiness

The audit passes configuration consistency and keeps `formal_result=false`, but
formal training is blocked until all of the following are implemented and
validated:

1. real context-conditioned FOMAML over the frozen task split;
2. qualified policy Modules executing through the full Continual Learning
   Pipeline;
3. a paired-world reference verifier for Knowledge evidence;
4. a feasibility/cost pilot for the draft horizon and budgets;
5. final review of thresholds, statistical analysis, and task difficulty.

The controlled discrete environment remains archived for regression only. No
Crafter value in this draft is a method-effectiveness result.

## Threshold review update (2026-10-06)

The read-only review at
`experiments/audit_crafter_thresholds.py` compares the candidate values with
the current Knowledge budget scan, verifier boundary audit, feasibility pilot,
and real-policy connection diagnostic. Under the current Beta/normal-bound
implementation, pure support and pure counterevidence reach their local
`n_min=10`, `tau_confirm=0.8`, or `tau_reject=0.2` decision after 13 valid
observations, within the candidate 200-unit Knowledge budget. The current real
paired-world evidence has only five valid counterevidence cases, so it remains
below `n_min` and does not confirm or reject a mechanism.

The review does not validate the formal `0.8` Module qualification threshold:
the diagnostic connection run used an explicitly non-formal `0.1` threshold,
and its observed qualification rates were `0.1--0.4`. The feasibility pilot
also produced no stable signal for stone, coal, or the three pickaxe tasks.
SPT validation batches and formal query-budget curves have not been run. The
review therefore keeps all numerical values as candidates and keeps formal
training blocked.

## Stone/coal horizon feasibility update (2026-10-06)

Because the CNN representation still produced no stone or coal successes at
horizon 256, a separate pilot fixed the CNN and 40-episode training budget
while increasing the horizon to 512. It is configured by
`configs/crafter_policy_horizon_pilot_v1.yaml` and stored in
`results/crafter_policy_horizon_pilot_9a78532/`.

Both `collect_stone` and `collect_coal` remained at `0/40` training success
and `0/10` independent qualification success. The audit in
`results/crafter_policy_horizon_audit_9a78532/` passes the non-formal boundary
checks and keeps formal training blocked. The result does not establish
unreachability; it shows that doubling the candidate horizon did not solve
the current policy learnability problem.

## Task prerequisite audit (2026-10-06)

The protocol audit
`experiments/audit_crafter_task_prerequisites.py` reads the Crafter 1.8.3
versioned collect/make rules and the candidate task order. Its report is
stored in `results/crafter_task_prerequisite_audit_f03c013_retry/`.

Crafter requires `wood_pickaxe: 1` to collect both stone and coal. The
candidate continual order places `obtain_wood_pickaxe` after those two tasks,
while the environment contract starts every episode with a fresh inventory.
The audit therefore reports direct prerequisite gaps for `collect_stone` and
`collect_coal`. `obtain_iron_pickaxe` also requires iron, for which the
candidate task list has no producer. These are task-protocol gaps under the
current fresh-episode design, not proofs that the resources are unreachable.
Formal training remains blocked until the task split, prerequisite handling,
and qualification protocol are revised and re-audited.

## Dependency-aware protocol candidate (2026-10-06)

The proposed revision is recorded separately in
`configs/crafter_prerequisite_aware_protocol_candidate_v1.yaml`. It keeps the
current RGB observation contract, changes the continual order to
wood → wood pickaxe → stone → stone pickaxe → coal/iron → iron pickaxe, raises
the wood and stone collection targets to cover the crafting costs, and adds an
explicit `collect_iron` producer. It marks persistent per-seed world state and
adapter support as pending implementation.

The audit at
`results/crafter_prerequisite_aware_candidate_audit_8fe5914/audit.json` finds
that inventory dependencies are ordered correctly, but table and furnace
setup are still undeclared. The candidate therefore remains non-formal and
fails readiness until world-object setup and state transfer are implemented
and independently validated. The existing six-task formal candidate is
unchanged.

## Dependency-aware manifest v2 (2026-10-08)

The first dependency-aware manifest left table and furnace as required objects
without declaring a producing task. The new candidate
`configs/crafter_prerequisite_aware_protocol_candidate_v2.yaml` keeps the
candidate non-formal and adds explicit `setup_table` and `setup_furnace` task
boundaries. Its order is wood -> table -> wood pickaxe -> stone -> stone
pickaxe -> coal/iron -> furnace -> iron pickaxe, so the audit can check both
inventory producers and setup producers. This is a manifest consistency repair,
not evidence that a learned policy can execute the full order. Persistent
session execution, independent setup qualification, budgets, and threshold
validation remain open. The v2 audit reports no inventory or world-object
producer gaps while retaining `formal_training_allowed=false`.

## Persistent session boundary diagnostic (2026-10-08)

The v2 setup order was exercised through
`CrafterContinualSession` with a verifier-generated public-action-consistent
route. The configuration is
`configs/crafter_continual_session_boundary_diagnostic_v1.yaml`, with output in
`results/crafter_continual_session_boundary_diagnostic_trace_v4/`.
After fixing the verifier's nearest-source greedy choice, all five seeds
produced a route through wood, table, wood pickaxe, four stone, and furnace.
All five public-action replays completed all five task boundaries, and the
compact public trace comparison found no first divergence. Inventory and setup
state were preserved at every boundary. This closes the current verifier and
session-boundary diagnostic for the selected route, but it remains a
verifier/action replay result rather than a learned-policy or formal result.
`begin_task()` now rejects a task whose required public setup has not been
confirmed, so this boundary cannot be satisfied by metadata alone.

## Continual session boundary implementation (2026-10-06)

`src/environments/crafter_continual.py` now defines a candidate
`CrafterContinualSession` that keeps one native world alive across ordered
task boundaries, plus explicit table/furnace setup and confirmation
contracts. `CrafterEnvironmentAdapter` records the current RGB observation so
`CrafterPolicyModuleExecutor` can opt into persistent execution with
`reset_before_execute=false`; its default remains fresh reset behavior.

The implementation is covered by the adapter and protocol regression tests,
but it does not perform world-object placement, persist a formal task state,
or authorize training. The dependency-aware candidate therefore remains
blocked until setup execution, state transfer, qualification data separation,
and the revised task manifest are validated together.

## Task-local composition candidate (2026-10-06)

The next candidate keeps the six primary targets unchanged and describes
their prerequisites as ephemeral same-episode plans in
`configs/crafter_task_local_composition_candidate_v1.yaml`. For example,
`collect_stone` keeps target `stone >= 1`, while its plan requests wood,
table setup, and wood pickaxe steps before the target step. The larger
quantities needed for crafting remain prerequisite capabilities rather than
new primary metrics.

`src/continual_learning/contracts.py` now provides `TaskPlan` and
`TaskPlanStep`; `InMemoryContinualLearningPipeline.run_task_plan` executes
the chain while recording each step's Skill Feedback and keeping runtime
evidence in the existing unknown Knowledge boundary. Capability compatibility
now treats inventory thresholds and world-object sets monotonically, so
`wood >= 3` satisfies a `wood >= 2` requirement and `{table, furnace}`
satisfies `{table}`.

The candidate plan audit at
`results/crafter_task_local_composition_audit_469d1bb_v2/` passes structural
checks, but world-object execution, auxiliary Module qualification, and
end-to-end Crafter validation remain pending. Formal training is still
blocked.

## Qualification feasibility update (2026-10-06)

The candidate-budget qualification pilot is implemented by
`experiments/run_crafter_qualification_feasibility.py` and configured by
`configs/crafter_qualification_feasibility_pilot_v1.yaml`. It trained one
isolated real RGB policy per task for 20 episodes and evaluated 20 independent
qualification episodes at the candidate 256-step horizon and formal `0.8`
success threshold. The run is stored in
`results/crafter_qualification_feasibility_pilot_eba9b2f_retry/`.

All six tasks had contract rate `1.0` and zero UNKNOWN qualification outcomes.
`collect_wood` reached `7/20` (`0.35`); `collect_stone`, `collect_coal`, and
all three pickaxe tasks reached `0/20`. No task formed a Module under the
formal gate. This is a feasibility result for the candidate budget, not a
method comparison or an environment-unreachability claim. The result keeps
formal training blocked and provides no basis for lowering the qualification
threshold; the current policy/training setup must first be made capable of
learning the candidate task set within an explicitly reviewed pilot.

## Task learnability budget update (2026-10-06)

The independent budget pilot is implemented by
`experiments/run_crafter_task_learnability_budget_pilot.py` and configured by
`configs/crafter_task_learnability_budget_pilot_v1.yaml`. It compares fresh
RGB policies trained for 20 and 80 episodes per task, then evaluates 20
independent qualification episodes at the same 256-step horizon and `0.8`
threshold. The result is stored in
`results/crafter_task_learnability_budget_pilot_cdfbc86/`.

At 20 episodes, `collect_wood` reached `5/20`; at 80 episodes it reached
`9/20`. Stone, coal, and all three pickaxe tasks reached `0/20` at both
budgets. Contract pass rate was `1.0` and UNKNOWN count was zero for every
task and budget. The larger budget therefore did not validate the candidate
qualification threshold or the full task split. This remains a non-formal
learnability observation, does not imply that any task is unreachable, and
keeps formal training blocked.

## RGB representation feasibility update (2026-10-06)

The representation pilot is implemented by
`experiments/run_crafter_policy_representation_pilot.py` and configured by
`configs/crafter_policy_representation_pilot_v1.yaml`. It compares the
reviewed `8x8` average-pool MLP with a small RGB CNN on independent wood,
stone, and coal tasks. Both use 40 training episodes and 10 qualification
episodes at horizon 256. The result is stored in
`results/crafter_policy_representation_pilot_391672a/`.

The CNN raised wood qualification from `3/10` to `6/10`, while both
representations scored `0/10` on stone and coal. No representation reached
the formal `0.8` qualification threshold. This is evidence that the current
RGB representation affects learnability, but it does not validate a new
formal policy contract or imply that stone or coal is unreachable. The audit
at `results/crafter_policy_representation_audit_391672a/` keeps formal
training blocked.

The machine-readable boundary check is
`experiments/audit_crafter_task_learnability_budget.py`; its report is stored
in `results/crafter_task_learnability_budget_audit_cdfbc86/` and keeps the
formal gate closed.

## First auxiliary prerequisite feasibility update (2026-10-06)

The first task-local prerequisite, `gather_wood_3`, was tested separately with
the candidate 256-step horizon and 20 independent qualification episodes. The
baseline 20-episode training run is stored in
`results/crafter_auxiliary_gather_wood3_pilot_3e623b9/`: training reached
`1/20`, qualification reached `0/20`, and the contract rate was `1.0`.

The training-only inventory progress reward was then added to the diagnostic
runner, while qualification remained unchanged and independent. The rerun at
`results/crafter_auxiliary_gather_wood3_pilot_eb490ee/` reached `0/20` in both
training and qualification. An 80-episode training comparison at
`results/crafter_auxiliary_gather_wood3_pilot_f491458/` also reached `0/20`
qualification. Finally, the average-pool MLP and CNN representation comparison
at `results/crafter_auxiliary_gather_wood3_representation_pilot_cabafba/`
reached `0/20` qualification for both representations.

These diagnostics show that the first auxiliary Module is not currently
qualified under the candidate budget. They do not justify lowering the gate or
claiming that wood collection is unreachable. The next feasibility decision is
whether to validate a longer horizon or redesign the RGB policy/training pilot;
downstream table and pickaxe Modules remain blocked until this prerequisite is
qualified.

The final direct horizon check used 512 steps with 40 training and 20
qualification episodes, stored in
`results/crafter_auxiliary_gather_wood3_horizon512_pilot_e10d900/`. Training
reached `1/40`, qualification remained `0/20`, and contract rate remained
`1.0`. Extending the horizon alone therefore did not produce a qualified
auxiliary Module. The current blocker is policy learnability under the RGB
interface, rather than a missing task prerequisite declaration or a short
horizon alone.

## Unit-capability composition update (2026-10-06)

The task-local candidate now represents the former `wood >= 3` prerequisite as
three repeated `wood >= 1` steps in one episode. The six formal primary targets
are unchanged; only ephemeral prerequisite composition changed. The fresh
structural audit at `results/crafter_task_local_composition_audit_ef654a1/`
passes.

The independent unit-capability pilot is stored in
`results/crafter_auxiliary_gather_wood1_pilot_5bbe3cf/`. With 40 training and
20 qualification episodes at horizon 256, training reached `10/40`, while
qualification reached `3/20` (`0.15`) with contract rate `1.0`. The unit split
improves training signal compared with `wood >= 3`, but it still does not form
a qualified Module under the candidate `0.8` gate. Downstream table and pickaxe
steps remain blocked pending a better RGB policy or training procedure.

The exact unit-capability representation comparison is stored in
`results/crafter_auxiliary_gather_wood1_representation_pilot_9520da1/`. At 40
training and 20 independent qualification episodes, the average-pool MLP
reached `7/20` qualification and the CNN reached `6/20`; both contract rates
were `1.0`, and neither reached the candidate `0.8` threshold. The CNN therefore
does not solve independent qualification for the actual repeated-plan unit
target. Static RGB representation is no longer the next diagnostic axis;
temporal context is the next candidate change.

The four-frame RGB diagnostic at
`results/crafter_auxiliary_gather_wood1_temporal_pilot_ce9cbcb/` compared the
existing average-pool MLP, CNN, and a four-frame average-pool stack on the exact
`gather_wood_1` target. Training success was `10/40`, `9/40`, and `19/40`,
respectively; independent qualification was `7/20`, `5/20`, and `7/20`.
Frame stacking improved training-side success but did not improve independent
qualification beyond the MLP baseline, so it does not produce a qualified
Module or justify changing the formal observation contract.

## PPO/GAE auxiliary training update (2026-10-06)

The non-formal PPO/GAE runner is configured by
`configs/crafter_auxiliary_gather_wood1_ppo_pilot_v1.yaml` and stores its result
under `results/crafter_auxiliary_gather_wood1_ppo_pilot_978a478/`. It keeps the
RGB average-pool input, target `wood >= 1`, 40 training episodes, and 20
independent qualification episodes. Training reached `15/40` and qualification
reached `10/20` (`0.50`), compared with the prior MLP REINFORCE result of
`7/20` qualification. PPO/GAE therefore improves this feasibility signal but
does not meet the candidate `0.8` gate or create a Module. The next budget
check can use PPO/GAE, while formal thresholds and the formal policy backend
remain unchanged.

An 80-episode PPO/GAE budget check is stored under
`results/crafter_auxiliary_gather_wood1_ppo_pilot_b70fa43/` and uses
`configs/crafter_auxiliary_gather_wood1_ppo_pilot_80_v1.yaml`. Training reached
`27/80` (`0.3375`), while the independent 20-episode qualification reached
`7/20` (`0.35`). This did not reproduce the 40-episode qualification signal;
the auxiliary Module remains unqualified, `module_registered` remains false,
and `formal_training_allowed` remains false. The result supports treating the
PPO improvement as a feasibility diagnostic rather than a stable estimate and
does not justify changing the formal threshold, observation contract, or
training budget.

## PPO/GAE multi-seed feasibility update (2026-10-06)

To check whether the 80-episode signal was seed-specific, the same non-formal
PPO/GAE candidate was run for seeds `1` through `4`, with independent training,
qualification, and action seeds. The outputs are stored in
`results/crafter_auxiliary_gather_wood1_ppo_multiseed_s1_8b2adce/` through
`results/crafter_auxiliary_gather_wood1_ppo_multiseed_s4_8b2adce/`; seed `0` is
the preceding result in
`results/crafter_auxiliary_gather_wood1_ppo_pilot_b70fa43/`.

Across the five seeds, training success was `0.3075 +/- 0.0376` (population
standard deviation), and independent qualification was `0.42 +/- 0.0678`.
Per-seed qualification rates were `0.35`, `0.40`, `0.55`, `0.40`, and `0.40`
for seeds `0` through `4`; none reached the candidate `0.8` gate. Every run
verified CUDA tensors, kept `formal_result=false`, and left Module registration
and Knowledge Evolution disabled. PPO/GAE therefore remains a useful
feasibility diagnostic but does not establish a qualified prerequisite Module
or permit formal training.

## Goal-action allowlist feasibility update (2026-10-06)

The PPO feasibility runner now supports an explicit task-local action allowlist
that is applied consistently during sampling and PPO updates. The diagnostic
configuration `configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_pilot_v1.yaml`
keeps only the public Crafter actions `noop`, four directional moves, `do`,
and `sleep` for `gather_wood_1`; it does not expose maps, coordinates, recipes,
or private state. This allowlist is a diagnostic task constraint and is not
part of the formal observation or policy contract.

The five-seed run is stored under
`results/crafter_auxiliary_gather_wood1_ppo_goal_actions_pilot_17ab5fe_s0/`
through `_s4/`. Training success was `0.6225 +/- 0.0634` (population standard
deviation), and independent qualification was
`0.75`, `0.80`, `0.60`, `0.55`, and `0.60`, for a mean of `0.66` and population
standard deviation `0.0970`. One of five seeds reached the candidate `0.8`
threshold; the runner still registered no Module and kept
`formal_result=false`. The allowlist improves the previous seed-0 qualification
from `0.35` to `0.75`, but the multi-seed result is not stable enough to change
formal thresholds or permit formal training. The next performance bottleneck
is RGB spatial and temporal credit assignment after exploration reduction.

## Paired temporal-stack feasibility update (2026-10-06)

Four-frame average-pool RGB input was added to the same PPO goal-action
allowlist runner. The paired configuration uses the same training and
qualification environment/action seed bases as the single-frame allowlist
comparison, with only `frame_stack=4` and the policy input dimension changed.
The result is stored in
`results/crafter_auxiliary_gather_wood1_ppo_goal_actions_temporal_pilot_28218f2_paired/`.

The temporal stack reached `51/80` training episodes (`0.6375`) and `14/20`
independent qualification episodes (`0.70`), compared with `44/80` (`0.55`)
and `15/20` (`0.75`) for the paired single-frame run. Temporal context improved
training-side success but reduced independent qualification by `0.05`; it did
not form a Module and remains `formal_result=false`. Four-frame stacking is
therefore not sufficient to clear the qualification gate. The next diagnostic
should target spatial credit assignment or explicit public progress features,
while keeping the formal RGB contract unchanged.

## Paired spatial-CNN feasibility update (2026-10-06)

A small two-layer CNN encoder was added to the PPO feasibility runner and
evaluated with the same single-frame RGB input, action allowlist, and seed bases
as the average-pool comparison. The configuration is
`configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_cnn_pilot_v1.yaml`,
with output in
`results/crafter_auxiliary_gather_wood1_ppo_goal_actions_cnn_pilot_89f3b86/`.

The CNN reached `52/80` training episodes (`0.65`) but only `10/20`
independent qualification episodes (`0.50`), compared with `44/80` (`0.55`)
and `15/20` (`0.75`) for the paired average-pool policy. The CNN therefore
improves training-side success while degrading held-out qualification in this
budget, does not register a Module, and remains `formal_result=false`. The
average-pool policy remains the current diagnostic baseline; neither temporal
stacking nor this small CNN justifies a formal observation-contract change.

## Training-only progress-reward feasibility update (2026-10-07)

The PPO runner was extended with the same public inventory increment shaping
used by the earlier REINFORCE diagnostic. A positive increase in the target
inventory contributes `progress_bonus=0.5` during training only; qualification
episodes use the native reward and terminal success bonus without this shaping.
The configuration is
`configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_progress_pilot_v1.yaml`,
and the result is stored in
`results/crafter_auxiliary_gather_wood1_ppo_goal_actions_progress_pilot_b75da37/`.

With the same seed bases and public action allowlist as the average-pool
baseline, training reached `53/80` (`0.6625`) while independent qualification
reached `13/20` (`0.65`). The paired no-progress run reached `44/80` (`0.55`)
and `15/20` (`0.75`). Progress shaping therefore raises training-side success
but lowers held-out qualification in this budget. It does not register a
Module, keeps `formal_result=false`, and does not justify changing the formal
reward contract or qualification threshold.

## Public-inventory observability isolation update (2026-10-07)

To separate RGB observability from optimization difficulty, the non-formal PPO
runner was given an optional auxiliary input containing the public `wood`
inventory value and an observed mask. Missing inventory remains represented as
unknown (`value=0`, `observed=0`); the diagnostic does not expose map, position,
recipe, or other private state. The configuration is
`configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_inventory_pilot_v1.yaml`,
and the result is stored in
`results/crafter_auxiliary_gather_wood1_ppo_goal_actions_inventory_pilot_5c6e394/`.

Using the same action allowlist and seed bases as the RGB-only comparison,
training reached `53/80` (`0.6625`) and independent qualification reached
`10/20` (`0.50`), versus `44/80` (`0.55`) and `15/20` (`0.75`) for the paired
RGB-only baseline. The auxiliary inventory feature therefore did not improve
held-out performance in this budget. It intentionally extends the formal RGB
contract, remains `formal_result=false`, and does not register a Module. The
diagnostic points to unstable spatial exploration and optimization rather than
inventory visibility alone; no observation-contract change is justified.

## Entropy-annealing feasibility update (2026-10-07)

The non-formal PPO runner was extended with an explicit linear entropy schedule
for training updates. The diagnostic starts at `entropy_coef=0.02` and ends at
`0.001` over the 80 training episodes, while qualification remains unchanged.
The configuration is
`configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_entropy_pilot_v1.yaml`,
and the result is stored in
`results/crafter_auxiliary_gather_wood1_ppo_goal_actions_entropy_pilot_454f510/`.

With the same RGB average-pool input, public action allowlist, and independent
seed split as the fixed-entropy baseline, training reached `54/80` (`0.675`)
and independent qualification reached `10/20` (`0.50`), compared with
`44/80` (`0.55`) and `15/20` (`0.75`) for the paired fixed-entropy run. Entropy
annealing therefore improves training-side success but reduces held-out
qualification. It remains a non-formal diagnostic, registers no Module, and
does not justify changing the formal optimizer schedule or qualification gate.

## Oracle-to-RGB imitation upper-bound update (2026-10-07)

To isolate RGB representation and spatial action learning from online
exploration, a non-formal behavior-cloning diagnostic used the existing Crafter
reference planner as a teacher. The teacher could inspect private world state
only to emit movement and interaction labels; the student received only the
reviewed RGB representation and the public action allowlist. The configuration
is `configs/crafter_rgb_oracle_imitation_diagnostic_v1.yaml`, with output in
`results/crafter_rgb_oracle_imitation_diagnostic_e8d549a/`.

All 10 teacher seeds reached `gather_wood_1` in 5--8 reference steps and
produced 64 RGB/action samples. After 40 supervised epochs, the RGB student
reached `0/10` on independent evaluation seeds. This is an upper-bound
diagnostic rather than a formal result: the teacher labels are oracle-side,
the sample set is intentionally small, no Module or Knowledge state was
updated, and `formal_result=false` is retained. The result indicates that the
current RGB encoder and data coverage cannot generalize the spatial action
mapping even when exploration is removed; further PPO tuning alone is unlikely
to clear the formal qualification gate.

An extended oracle-data check used 100 teacher seeds and 607 RGB/action samples
with the same student architecture, stored in
`results/crafter_rgb_oracle_imitation_extended_diagnostic_1755723/` and
configured by `configs/crafter_rgb_oracle_imitation_extended_diagnostic_v1.yaml`.
The teacher again succeeded on `100/100` worlds, while the RGB student reached
`4/20` (`0.20`) on disjoint evaluation seeds. More teacher coverage improves
over the small diagnostic (`0/10`), so data coverage contributes to the
bottleneck, but the result remains far below the candidate `0.8` qualification
gate. It still registers no Module, keeps `formal_result=false`, and does not
justify formal training; the next work should target a representation or
training method designed for spatial generalization rather than isolated PPO
hyperparameter changes.

## Spatial imitation and action-balance feasibility update (2026-10-07)

The next diagnostic separated spatial representation from online exploration by
replacing the RGB student's average-pool MLP with a two-layer spatial CNN and
using inverse-square-root class weights in the supervised action loss. The
configuration is
`configs/crafter_rgb_oracle_imitation_spatial_v1.yaml`, with output in
`results/crafter_rgb_oracle_imitation_spatial_205a7b4/`.

The oracle teacher again succeeded on all `100/100` worlds and produced `607`
RGB/action pairs. Six public actions were represented, with counts
`[100, 176, 76, 87, 68, 100]` for action IDs `0` through `5`; no private
state entered the student. The spatial CNN student reached `0/20` on disjoint
evaluation worlds, compared with `4/20` for the previous average-pool MLP
student trained on the same extended teacher set. The experiment remains
`formal_result=false`, registers no Module, and performs no Knowledge update.

This result does not support changing the formal RGB contract or claiming a
spatial CNN solution. It also shows that representation and class weighting
must be evaluated separately; the next diagnostic should keep one variable
fixed while testing the other, before any formal feasibility decision.

The uniform-loss control used the same spatial CNN configuration, teacher
seeds, evaluation seeds, and RGB/action generation, with ordinary
cross-entropy instead of inverse-square-root weighting. It is stored in
`results/crafter_rgb_oracle_imitation_spatial_uniform_760911f/` and also
reached `0/20`. Those two historical runs were made before model
initialization was seeded, so they are not an exact initialization-paired
ablation; they provide a consistent warning signal, not a causal conclusion
about the loss weighting. The RGB spatial action-generalization gate remains
open, and formal training remains disallowed.

The same audit found that this is not only a memory problem. Among the 100
teacher episodes, the first movement observation had 85 exact RGB values, and
three repeated values were paired with different movement labels. At those
states the nearest tree is outside the reviewed local view, so the label is
determined by the oracle's private global map. No deterministic feed-forward or
recurrent policy can recover that hidden direction from the identical initial
observation alone. This is an identifiability failure for the current
oracle-to-RGB target, not evidence that a larger CNN would solve the task.

The feasible protocol options are therefore: constrain a diagnostic task so a
target resource is locally visible at the decision point; define the teacher
around a public-observation exploration policy instead of the private
global-nearest-tree route; or explicitly add a reviewed public goal signal to
the observation contract. Each option changes the task or interface and must
be re-audited and frozen before formal training. Keeping the current hidden
target labels while increasing model size or horizon does not close this gate.

## Diagnostic boundary audit (2026-10-07)

The extended oracle imitation diagnostic exposed two methodological issues that
must be separated from RGB learnability. The original recording teacher used
the oracle planner's private `_face` operation, which directly changed the
player's facing without a Crafter action. Across 100 wood episodes this
occurred in 20 of 100 facing events, so the corresponding interaction labels
were not all executable from the preceding learner observation.

The diagnostic now has a `public_action_consistent` teacher mode. It changes
facing through a legal `move_*` action and moves the subsequent interaction
observation after that action. The corrected run is stored in
`results/crafter_rgb_oracle_imitation_public_teacher_e0c6db1/`. The teacher
still succeeds on `100/100` worlds and supplies `622` samples; the average-pool
student reaches `3/20`, close to the previous `4/20` result. The hidden facing
mutation was therefore a real boundary defect, but not the sole performance
bottleneck.

The same revision moves the diagnostic seed before policy construction. Older
CNN results remain valid non-formal observations but are not treated as strict
reproducibility or paired-loss evidence.

The corrected run also reports `584` unique exact RGB observations and four
exact observations with conflicting action labels. This is direct evidence of
single-frame state aliasing: the same reviewed RGB can require different
movement actions because the oracle target and global route are outside the
observation. A feed-forward RGB classifier cannot be treated as a sufficient
state policy for this navigation target. The result remains non-formal and
does not authorize changing the observation contract or starting formal
training.

## No-teacher collect-wood FOMAML feasibility update (2026-10-08)

To remove the oracle-teacher confound, a separate non-formal pilot used only
the real RGB policy, context-conditioned FOMAML, and the independent
`collect_wood` prerequisite. The configuration is
`configs/crafter_collect_wood_fomaml_pilot_v1.yaml`, with five independent
replicas in
`results/crafter_collect_wood_fomaml_pilot_253b557/`. Stone, coal, and craft
targets were intentionally excluded because their task-local prerequisites are
not yet validated.

The three independent query repeats per replica gave mean success `0.280` for
the active policy and `0.267` after the candidate update. Per-replica changes
were `0.000`, `+0.067`, `0.000`, `+0.067`, and `-0.200`. Every replica used
CUDA tensors, disjoint train/support/query seeds, an unchanged active SPT, and
a changed candidate policy/template. The candidate update therefore executed
through the real FOMAML path but did not show stable query improvement, form a
Module, or authorize formal training.

## Public world-object setup boundary update (2026-10-08)

The adapter now confirms Crafter table/furnace setup from a public transition
only: a `place_table` or `place_furnace` action must produce the exact
versioned inventory decrement. It does not inspect semantic maps, player
coordinates, achievements, or RGB pixels for setup confirmation. The boundary
diagnostic is configured by
`configs/crafter_setup_boundary_diagnostic_v1.yaml` and stored in
`results/crafter_setup_boundary_diagnostic_c21379d/`.

The oracle-side executable diagnostic completed table and furnace setup on
`3/5` seeds, and all completed seeds reported both public setup capabilities.
The other two seeds had no path to a stone source before furnace placement;
they are reachability failures in the verifier route, not setup confirmation
failures. The run remains `formal_result=false`, uses private world state only
on the verifier side, and performs no Module or Knowledge update. Full
prerequisite task feasibility, persistent state transfer, and independent
qualification remain open.

## Task-local Pipeline boundary update (2026-10-08)

The task-local plan was next exercised end to end with the real
`CrafterTaskPlanExecutor` and `InMemoryContinualLearningPipeline`. The
verifier generated a public-action-consistent Crafter action script in a
separate shadow world, then the learner-facing RGB/inventory adapter replayed
that script with fixture-qualified Modules. The diagnostic is configured by
`configs/crafter_task_local_oracle_pipeline_diagnostic_v1.yaml` and stored in
`results/crafter_task_local_oracle_pipeline_diagnostic_fix2/`.

All five seeds completed the `obtain_wood_pickaxe` plan: three accumulated wood
steps, public table setup, and the final wood-pickaxe target. The three repeated
`wood >= 1` requests correctly reused the same qualified fixture Module three
times; the executor now advances its internal scripted progress by invocation,
which makes the same-episode resource accumulation explicit. Each episode
returned five Skill Feedback records and five ordinary Knowledge Evidence
records with `evidence_validity=unknown`. The verifier action script and fixture
Modules do not establish policy learnability, qualification, or method
effectiveness; `formal_result=false` remains in force. Independent qualification
of real learned Modules and the stone/coal/furnace prerequisite chains are
still open.

The same diagnostic was extended to the `collect_stone` task using a
verifier-only wood, table, wood-pickaxe, and stone action script. All five
seeds completed the six-step candidate plan, including the `wood_pickaxe`
prerequisite capability and the final stone target. The final stone request
received the accumulated `wood_pickaxe` capability, while ordinary runtime
evidence remained `unknown`. The output is stored in
`results/crafter_collect_stone_oracle_pipeline_diagnostic_fix1/`. This confirms
candidate capability routing and public state transfer for one additional
chain; it does not validate a learned Module or establish that the policy can
discover the chain.

## Persistent RGB Policy feasibility pilot (2026-10-08)

The next non-formal pilot used the real RGB Policy on one persistent Crafter
world per seed, with no teacher, Policy update, FOMAML step, Module, Knowledge
update, or SPT switch. The runner is
`experiments/run_crafter_persistent_policy_feasibility.py`, configured by
`configs/crafter_persistent_policy_feasibility_pilot_v1.yaml`, and the output
is in `results/crafter_persistent_policy_feasibility_pilot_d472815/`.

The validated five-task chain was `collect_wood`, `setup_table`,
`obtain_wood_pickaxe`, `collect_stone`, and `setup_furnace`. All five seeds
failed the first task before reaching the target (`0/25` task completions).
Four seeds terminated after `92--207` environment steps and one exhausted its
256-step task budget; one seed had one wood but no three-wood target. The public boundary audit still reported
continuous step, inventory, setup, and episode state across every attempted
task boundary. This isolates the current performance blocker as basic RGB
navigation/survival/resource acquisition under the current untrained Policy,
not a session-state transfer failure. The result remains
`formal_result=false` and does not justify changing the observation contract
or starting formal training.

During pilot preparation, the v2 prerequisite manifest was corrected so
`obtain_wood_pickaxe` requires `wood: 1` at its task boundary after
`setup_table` consumes two wood; `wood: 3` remains the cumulative chain cost,
not the immediate boundary inventory requirement.
