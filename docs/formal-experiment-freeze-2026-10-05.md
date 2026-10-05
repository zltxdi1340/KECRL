# Formal Controlled Experiment Design Freeze

This freeze records the executable budget and acceptance rules for the next
controlled discrete-resource experiment. It does not promote any result to a
paper result: the configuration and all future outputs must keep
`formal_result=false` until the runner and statistical review gates pass.

The machine-readable source is
`configs/controlled_torch_fomaml_formal_freeze_v1.yaml`. Its read-only audit is
`experiments/audit_formal_freeze.py`; a passing audit means that the manifest
and configuration are internally consistent and ready for runner
implementation, not that training may start.

The frozen scope is `controlled_discrete_resource_v1` with the existing stage
manifest, seeds `0..4`, four variants, and 200 episodes per role and seed.
Roles are `train`, `support`, `query`, `qualification`, and `spt_validation`.
Role IDs must be disjoint within every seed and every variant must use the same
manifest and role assignments.

Qualification is a per-task/SPI gate. Each task receives 20 held-out
qualification episodes, requires success rate at least `0.8` and contract pass
rate `1.0`, and reports unknown/invalid outcomes separately. A Module can be
registered only after every task gate required by the run passes; the runner
must create the actual `SPI`, contract, `Module`, and Skill Library record.

SPT acceptance compares active and candidate versions on the same SPI set,
support budget, and two independent validation batches of 10 episodes per task.
Each batch must meet at least 10% query-learning-efficiency improvement, keep
each existing SPI's success-rate regression at or below `0.05`, and have no
hard contract violation. Insufficient evidence is `inconclusive` and keeps the
active pointer. Accepted candidates preserve `candidate`, `active`, and
`previous_stable` versions without rewriting existing Modules.

The primary metric is independent query learning efficiency at support
checkpoints `0, 50, 100, 200`; unreached thresholds remain right-censored.
Secondary metrics include query success/loss, qualification and reuse rates,
task outcomes, Knowledge status accuracy, runtime, and peak GPU memory.

The runner must require CUDA rather than silently falling back to CPU, use one
GPU, and write the complete provenance set listed in the configuration. The
candidate runner now enforces strict CUDA, evaluates qualification per task,
creates SPI/contract/Module records only for task gates that pass, and writes
the frozen support checkpoints. Module reuse through the in-memory Skill
Library remains an explicit implementation metric to wire into the task
execution path before promotion to a paper result. The contract check still
uses the controlled environment's finite-horizon execution contract and must
be audited against a richer adapter before external-environment claims.
