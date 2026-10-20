# Crafter Split Feasibility Pilot (2026-10-05)

This pilot checks whether the candidate Crafter task split and one-step
context-conditioned FOMAML path are executable at a small scale. It is a
feasibility diagnostic, not a formal comparison or paper result.

## Provenance and scope

- Runner: `experiments/run_crafter_split_feasibility.py`
- Config: `configs/crafter_split_feasibility_pilot_v1.yaml`
- Commit: `b73ddaaa8cd087b716c6f98cb15f1c8bee5784dc`
- Device: one visible RTX 4090 (`CUDA_VISIBLE_DEVICES=0`)
- Replicas: seeds `0..4`
- Episode horizon: 256 steps
- Evaluation query repeats: 3 per task
- Training tasks: `collect_wood`, `collect_stone`
- Evaluation tasks: the two training tasks plus `collect_coal` and three
  pickaxe targets
- Formal-result flag: `false`

The pilot uses the existing Crafter adapter, RGB feature conversion, and
context-conditioned FOMAML implementation. It verifies CUDA tensors, disjoint
training/evaluation seeds, disjoint support/query seeds, active-policy
isolation, and candidate-only parameter updates. It does not update Knowledge,
switch an SPT pointer, qualify/register a Module, or run the Continual Learning
Pipeline.

## Diagnostic values

Across the five replicas and 90 evaluation query episodes per phase:

| phase | query success |
|---|---:|
| active before candidate update | 4/90 = 0.0444 |
| candidate after one meta update | 3/90 = 0.0333 |

By task, the only nonzero success was `collect_wood`: mean success was `0.2667`
before and `0.2000` after the candidate update. `collect_stone`,
`collect_coal`, and all three pickaxe targets were `0.0` in both phases.

## Interpretation

The adapter, CUDA execution, seed split, context-conditioned candidate update,
and active/candidate isolation are operational. The proposed small budget does
not yet provide a useful learning-efficiency signal: the candidate update did
not improve the held-out query rate, and the craft transfer tasks did not
produce a success in this pilot. This is evidence that the current policy,
reward, horizon, and one-update budget need feasibility work; it is not an
environment impossibility claim and it does not justify changing formal
thresholds from this single pilot.

## Required work before formal training

1. connect qualified policy Modules to the full Pipeline;
2. implement the paired-world Knowledge reference verifier;
3. test longer or staged training budgets and task-specific feasibility without
   changing the formal candidate configuration silently;
4. decide whether the transfer craft tasks belong in the formal comparison only
   after they are learnable under a predeclared budget;
5. rerun the freeze audit after those changes, keeping `formal_result=false`.
