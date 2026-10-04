
﻿# Abstract

Continual reinforcement learning systems must update environment knowledge and
skill-generation procedures without conflating structural evidence with
ordinary execution feedback. We present KECRL, a layered design that separates
a Knowledge Bank for environment mechanisms from a Skill Library containing
SPTs, SPIs, and qualified Modules. A continual-learning pipeline coordinates
mechanism retrieval, implementation requests, execution, transition
inspection, and two explicitly separated feedback channels. Knowledge
Evolution uses scoped support, counterevidence, and UNKNOWN outcomes, while
Skill Evolution uses support/query-separated context-conditioned FOMAML and
candidate SPT versioning.

We implement and exercise the design in a controlled discrete-resource
environment on a CUDA PyTorch policy backend. The candidate stage includes a
Module-reuse baseline, separate Knowledge and Skill ablations, five seeds, and
independent query and qualification roles. The runs validate end-to-end
execution, data-contract boundaries, result provenance, and CUDA tensor use.
They do not establish a final performance advantage: the controlled results
remain marked `formal_result=false`, the diagnostic paired interval crosses
zero, and external-environment validation is pending. The artifacts therefore
serve as an auditable implementation stage toward formal continual-learning
experiments rather than as a completed benchmark claim.
