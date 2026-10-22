
## KECRL

Minimal CPU smoke and server GPU workflow. The smoke configuration is deliberately a controlled interface validation, not a research result.

```powershell
python -m experiments.run --config configs/smoke_cpu.yaml
python -m experiments.evaluate results/smoke/result.json
python scripts/check_environment.py
```

On a GPU server, install the environment-specific framework separately, run `python scripts/check_environment.py`, then use `configs/server_gpu.yaml`. `device: cuda` fails explicitly when CUDA is unavailable; `device: auto` selects CUDA only when available. Results contain configuration, seed, runtime metadata, JSON and CSV summaries. Formal environment, baseline, metrics and training budgets remain experiment decisions and must be recorded before formal runs.

## Current research direction (2026-10-20)

The repository now follows a staged validation route:

1. **LIBERO** for same-family SPT/SPI instance transfer and meta-learning efficiency;
2. **Continual World** for sequential task streams, forward transfer and forgetting;
3. **Crafter** for long-term memory, exploration and Knowledge Evolution;
4. **RoboCasa365** for the final controlled Knowledge--Skill joint evaluation.

The detailed Chinese route is recorded in [the research roadmap](docs/research-roadmap-2026-10-20.md). The LIBERO installation, pinned dataset audit, and KECRL adapter smoke are recorded in [the environment feasibility note](docs/libero-environment-feasibility-2026-10-21.md); the candidate Skill Family, trajectory structure, leakage-audited task/data roles, CUDA runtime, executable BC/SPT/SPI path, first source-SPI learning result, and closed-loop attribution are recorded in [the spatial family audit](docs/libero-spatial-skill-family-audit-2026-10-21.md), [the demonstration phase audit](docs/libero-demonstration-phase-structure-audit-2026-10-21.md), [the transfer split](docs/libero-spatial-transfer-split-2026-10-21.md), [the CUDA smoke](docs/libero-cuda-training-path-smoke-2026-10-21.md), [the BC Policy pipeline check](docs/libero-bc-policy-pipeline-check-2026-10-21.md), [the single-SPI BC diagnostic](docs/libero-single-spi-bc-learnability-diagnostic-2026-10-21.md), and [the closed-loop attribution](docs/libero-bc-closed-loop-attribution-2026-10-21.md). The attribution found that stored same-index observations occur after their paired actions. The matched 500-update same-index/causal pre-action comparison, frozen-weight recurrent-state follow-up, and six-start paired action/state divergence trace are complete. The latter found measurable translation-action error and growing joint-state drift before the first gripper sign mismatch; it remains a one-seed behavior diagnostic and does not identify a root cause. Protocols and results are recorded in [the causal-alignment diagnostic note](docs/libero-bc-causal-alignment-diagnostic-2026-10-22.md) and [the hidden-state diagnostic note](docs/libero-bc-hidden-state-diagnostic-2026-10-22.md). The next diagnostic will measure immediate one-step state effects from matched expert states. None of these checks is an SPT/FOMAML comparison or Module qualification. Historical controlled-environment and Crafter materials are preserved in the [archive](docs/archive/README.md). The robotics benchmark survey is in [robotic manipulation and continual task streams](docs/literature/robotic-manipulation-continual-task-streams-2026-10-20.md).

## Crafter historical status

The Crafter work remains a diagnostic and Knowledge Evolution line; no wood Module has passed its complete qualification contract. Read the [archive index](docs/archive/README.md) before interpreting historical reports. Raw RGB traces, features, checkpoints and repeated run directories remain on the experiment host rather than in the source archive.

The previous Crafter SPT/SPI transfer pilots, PPO calibration, RGB audits and failure-attribution reports are historical calibration and remain available under `docs/archive/crafter/`. They do not qualify a natural Crafter Module or validate full KECRL.
