
﻿# Introduction

Continual reinforcement learning agents face two related but distinct sources
of change. They must infer how an environment is structured, and they must
learn executable skills that satisfy contracts under changing task contexts.
Treating both as one undifferentiated memory creates a provenance problem: a
successful policy execution does not by itself establish a structural
mechanism, while a mechanism hypothesis does not identify a qualified reusable
implementation.

KECRL addresses this problem with an explicit layered architecture. The
Knowledge Bank stores scoped environment mechanisms and structural evidence.
The Skill Library stores SPTs, SPIs, and qualified Modules together with
skill-side experience and version metadata. A continual-learning pipeline
connects the two through mechanism retrieval, implementation requests,
execution, transition inspection, and separate Knowledge Evidence and Skill
Feedback. The separation preserves the semantics of `unknown` and
`unavailable`, prevents policy-side state from entering structural knowledge,
and makes candidate SPT updates auditable without mutating active versions or
existing Modules.

The paper makes three contributions. First, it specifies the object and
information boundaries needed to evolve environment knowledge and skills
independently. Second, it gives minimal update and qualification procedures:
Beta-Binomial evidence updates for atomic mechanisms, support/query-separated
context-conditioned FOMAML for skill evolution, and versioned SPT candidate
acceptance. Third, it provides an executable CUDA-controlled stage with
disjoint data roles, a reusable Module path, component ablations, and
provenance-preserving analysis.

Our current evidence is deliberately scoped. The controlled stage verifies the
interfaces, feedback routing, qualification path, and reproducible execution
of the layered system. It is not presented as a final benchmark result: the
environment is deterministic, the stage Policy and FOMAML paths are limited
implementations, the seed count is small, and Crafter validation remains
future work. This distinction lets the implementation evidence remain useful
without overstating what the current experiments establish.
