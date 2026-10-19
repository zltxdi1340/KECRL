
## KECRL

Minimal CPU smoke and server GPU workflow. The smoke configuration is deliberately a controlled interface validation, not a research result.

```powershell
python -m experiments.run --config configs/smoke_cpu.yaml
python -m experiments.evaluate results/smoke/result.json
python scripts/check_environment.py
```

On a GPU server, install the environment-specific framework separately, run `python scripts/check_environment.py`, then use `configs/server_gpu.yaml`. `device: cuda` fails explicitly when CUDA is unavailable; `device: auto` selects CUDA only when available. Results contain configuration, seed, runtime metadata, JSON and CSV summaries. Formal environment, baseline, metrics and training budgets remain experiment decisions and must be recorded before formal runs.

## Crafter feasibility status

The current Crafter work is still diagnostic; no wood Module has passed its complete qualification contract. Read the [current Chinese status report](docs/crafter-diagnostic-status-2026-10-15.md) and [episode-isolated coverage results](docs/crafter-wood3-spatial-coverage-2026-10-15.md) before interpreting the archived summaries. The raw RGB traces, features, and checkpoints remain on the experiment host.

The current research question follows the [SPT/SPI transfer protocol](docs/crafter-spt-spi-transfer-protocol-2026-10-15.md): compare template updates with ordinary pretraining on held-out gather bindings. The non-formal [E1 integration pilot](docs/crafter-gather-spt-transfer-pilot-2026-10-15.md) is historical calibration. E1/E2 v2 CUDA training and corrected full RGB audits have completed. The disclosed post-training audit correction found lower E2 AUC for FOMAML than ordinary pretraining; a subsequent geometry audit identified uneven material positions across role pools. Read the [current run report](docs/crafter-spt-spi-formal-run-2026-10-18.md) for results, limitations, and the next diagnostic plan. This does not qualify a natural Crafter Module or validate full KECRL.
