# Crafter Protocol Audit (2026-10-05)

This audit records the candidate episode split after making verifier-only world
seed derivation deterministic. It is preparation work, not a formal training
run.

## Provenance

- Config: `configs/crafter_task_protocol_v1.yaml`
- Protocol: `crafter_task_protocol_v1`
- Crafter version target: `1.8.3`
- Seeds: `0..4`
- Commit: `a97719fd8276f4628a93a076ff9f5d4e5bd664f7`
- Result directory: `results/crafter_protocol_preview_a97719f/`
- Formal-result flag: `false`

The generated manifest contains 420 assignments:

| role | assignments |
|---|---:|
| train | 60 |
| support | 60 |
| query | 60 |
| qualification | 60 |
| SPT validation | 120 |
| evaluation | 60 |

SPT validation has separate support and query phases. Episode IDs and
environment seeds are unique, every episode uses a fresh native reset index,
and the verifier-only derived world seed is now SHA-256 based rather than
Python's process-randomized `hash()`.

## Readiness result

The split audit passed its internal checks, but
`ready_for_formal_training=false`. The remaining blockers are:

- final candidate task families and held-out SPI split;
- formal budgets, reward definition, and episode horizon;
- real policy FOMAML integration for this task split;
- qualified policy Module execution through the full Pipeline;
- Knowledge intervention protocol and reference verifier;
- qualification, SPT acceptance, and statistical thresholds.

Unique seeds establish reproducible assignment provenance; they do not by
themselves prove distinct layouts, held-out SPI generalization, or method
effectiveness. No training or formal comparison was started from this audit.
