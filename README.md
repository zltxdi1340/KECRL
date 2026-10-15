
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

The next research priority is the [SPT/SPI transfer protocol](docs/crafter-spt-spi-transfer-protocol-2026-10-15.md): compare template updates with ordinary pretraining on held-out gather bindings. The CNN/meta-learning connection and multi-material fixture now have a non-formal [E1 integration pilot](docs/crafter-gather-spt-transfer-pilot-2026-10-15.md); its [design manifest](configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml) remains non-frozen, and the formal E1/E2 experiment has not started.
