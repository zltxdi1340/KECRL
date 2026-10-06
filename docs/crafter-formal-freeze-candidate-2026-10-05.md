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
