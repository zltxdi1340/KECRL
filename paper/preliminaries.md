
﻿# Preliminaries

## Continual task interaction

At task time `t`, the agent observes a public interface `h(s_t)`, selects
actions from `A`, and receives transition information that may be
partial or uncertain. A task result is classified as `completed` only when the
goal is detected from actual public state. `continued` means the task remains
active, `unavailable` means no compatible implementation can be provided under
the declared contract, and `unknown` means that the required state or outcome
cannot be determined. These labels are not interchangeable failure codes.

## Capabilities and mechanisms

A Capability is an agent-independent predicate over publicly detectable state.
The initial Capability Schema `V_0` is reviewed and frozen. A mechanism
`M=(P_start,P_hold,Y)` relates startup conditions, conditions that must
remain valid during execution, and a target Capability. Conditions within one
mechanism are conjunctive; alternative mechanisms may reach the same target.

Knowledge Evidence is scoped public environmental evidence collected under a
declared intervention and validity rule. Support, structural counterevidence,
and UNKNOWN have different statistical meanings. Ordinary Policy success or
failure is not automatically structural evidence.

## Skill objects

An SPT describes how a Skill Family generates and adapts SPI specifications. An
SPI binds an SPT to a task context, contract, implementation specification,
and adaptation requirements; it is a learning specification rather than an
executable skill. A Module is a qualified executable implementation bound to a
specific SPI, Policy, contract, scope, and provenance metadata. Existing
Modules may be reused only when the request is contract-, scope-, and resource-
compatible.

The Knowledge Bank stores mechanisms and Knowledge Evidence. The Skill Library
stores SPTs, SPIs, qualified Modules, and skill-side feedback. Policy
parameters, gradients, complete trajectories, and Module performance do not
cross into the Knowledge Bank.
