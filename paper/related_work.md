
﻿# Related Work

## Continual reinforcement learning

Continual reinforcement learning studies adaptation across a sequence of
tasks, environments, or data distributions, with taxonomies that distinguish
the scope and driver of non-stationarity \cite{Khetarpal2020ContinualRL}.
KECRL focuses on a specific
systems problem within this setting: separating persistent environment
structure from persistent skill-generation state so that each can evolve under
its own evidence and qualification rules. The paper should be positioned
against representative continual-learning and transfer-learning baselines in
the final submission; citation selection remains pending the literature pass.

## Meta-learning and skill reuse

Gradient-based meta-learning provides a natural basis for adapting a skill
initializer to new task contexts. MAML trains an initialization for rapid task
adaptation \cite{Finn2017MAML}, while first-order methods remove inner-loop
second-order terms \cite{Nichol2018FirstOrder}. KECRL uses a context-conditioned FOMAML path
as its reference Skill Evolution procedure, while treating support, query,
qualification, and SPT validation as distinct data roles. This separation is
central to the claimed traceability and is independent of any particular
meta-learning optimizer.

Unsupervised skill discovery methods such as DIAYN learn diverse reusable
behaviors without task rewards \cite{Eysenbach2019DIAYN}, whereas the KECRL
Module is created only after a task-conditioned implementation satisfies an
explicit contract and qualification test. Option-Critic learns option policies
and termination conditions jointly \cite{Bacon2017OptionCritic}; KECRL instead
separates the executable Module lifecycle from structural Knowledge Evolution.

## Structured knowledge and model-based control

Model-based and structured RL methods use abstractions, affordances, options,
or learned transition structure to improve planning. The options framework
formalizes temporally extended actions between MDP and semi-MDP descriptions
\cite{Sutton1999Options}; successor features separate transferable dynamics
features from task rewards and support policy reuse \cite{Barreto2017SuccessorFeatures}.
KECRL distinguishes
scoped Knowledge Evidence from executable skill state: a mechanism claim is
updated only by valid public structural evidence, whereas a Module is created
only after skill-side qualification. This prevents ordinary policy outcomes
from silently becoming environment facts.

Model-based agents such as Dreamer learn latent dynamics and improve behavior
through imagined trajectories \cite{Hafner2020Dreamer}. KECRL does not claim a
world-model contribution in the current stage; its focus is the evidence and
version boundary between environment mechanisms and reusable skills.

Modular transfer architectures such as Progressive Neural Networks preserve
previous task columns while adding lateral transfer pathways
\cite{Rusu2016Progressive}. KECRL shares the concern with retention and reuse,
but represents persistence as versioned SPT/SPI/Module objects and qualified
contracts rather than as a single growing policy network.

## Positioning and citation status

The initial reference set covers continual RL, gradient-based meta-learning,
temporal abstraction, successor-feature transfer, skill discovery, model-based
RL, and modular transfer. Each citation is used only for the scope stated
above; broader benchmark comparisons remain future work.
