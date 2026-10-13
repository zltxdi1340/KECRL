
## KECRL

Minimal CPU smoke and server GPU workflow. The smoke configuration is deliberately a controlled interface validation, not a research result.

```powershell
python -m experiments.run --config configs/smoke_cpu.yaml
python -m experiments.evaluate results/smoke/result.json
python scripts/check_environment.py
```

On a GPU server, install the environment-specific framework separately, run `python scripts/check_environment.py`, then use `configs/server_gpu.yaml`. `device: cuda` fails explicitly when CUDA is unavailable; `device: auto` selects CUDA only when available. Results contain configuration, seed, runtime metadata, JSON and CSV summaries. Formal environment, baseline, metrics and training budgets remain experiment decisions and must be recorded before formal runs.

## Crafter feasibility status

The current Crafter work is still diagnostic; no wood Module has passed its complete qualification contract. Read the [Chinese status report](docs/crafter-diagnostic-status-2026-10-13.md), [RGB action-chain audit](docs/crafter-wood3-rgb-chain-audit-2026-10-13.md), [single-seed learning pilot](docs/crafter-wood3-rgb-chain-learning-diagnostic-2026-10-13.md), and [same-state cross-readout audit](docs/crafter-wood3-rgb-chain-cross-readout-2026-10-13.md) before interpreting the archived summaries. The raw RGB traces, features, and checkpoints remain on the experiment host.
