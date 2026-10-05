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
