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
