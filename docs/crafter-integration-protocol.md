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

Next: freeze task/observation contracts and independent episode roles, connect
real Policy initialization to context-conditioned FOMAML, then connect qualified
Modules to actual policy execution. Baseline retains Module reuse with both
Evolution components off; ablations switch each separately. Seeds 0-4 are the
user-selected candidate set. Crafter budgets, qualification thresholds, SPT
acceptance/non-regression values, and formal statistics remain pending.
