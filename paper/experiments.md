
# Experiments: Controlled Stage Candidate

> Draft status: candidate-stage report. The results below are retained as
> `formal_result=false` and must not be presented as final paper evidence.

## Setting

The first executable stage uses the deterministic controlled discrete-resource
environment with five disjoint roles per seed: train, support, independent
query, qualification, and SPT validation. The candidate run uses seeds
`0,1,2,3,4`, 200 episodes per role, one NVIDIA RTX 4090 (GPU1), Python 3.10.22,
PyTorch 2.14.1+cu130, and commit `c5e15c4`. The policy backend is a CUDA
PyTorch categorical REINFORCE implementation; context-conditioned FOMAML is
the current reference path for Skill Evolution.

The comparison contains the full method, a Module-reuse baseline with both
evolution components disabled, a Knowledge Evolution ablation, and a Skill
Evolution ablation. A Knowledge action prior of strength 1.5 is enabled only
when Knowledge Evolution is enabled. Qualification requires 20 held-out
episodes, success rate at least 0.8, and zero hard contract violations.

## Candidate-stage results

The full artifact is [the generated main-results table](../results/formal_stage_v1/tables/main_results.md).
Mean query success was 0.995 for the method and Skill ablation and 0.991 for
the baseline and Knowledge ablation. The method's paired query-success delta
against the baseline was 0.004, with a five-seed diagnostic bootstrap interval
of [-0.002, 0.012]. Every variant qualified and every run reached the query
threshold. Knowledge status accuracy was 1.0 for Knowledge-enabled variants;
ordinary query execution evidence remained `unknown`, while Skill Feedback was
recorded separately.

The support curve is reported at 0, 50, 100, and 200 support episodes. For
the method, mean query success was 0.977, 0.963, 0.981, and 0.996 at these
budgets. These values describe the current controlled backend and are useful
for checking reproducibility and data flow. They do not establish a general
learning-efficiency advantage.

## Scope and limitations

This stage verifies that the Knowledge Bank, Skill Library, Module
qualification, CUDA policy execution, continual-learning pipeline, and result
provenance can be exercised together. It does not yet establish performance in
Crafter or another external environment. The Policy and FOMAML components are
stage implementations, and the five-seed bootstrap interval is diagnostic.
The action prior has shown sensitivity in holdout scans, including interactions
with Skill Evolution, so its value remains a candidate setting. Formal paper
claims require a frozen backend, final seed budget, independent statistical
procedure, and external-environment validation.
