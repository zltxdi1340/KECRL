# Repository Guidance

## Project Context

- KECRL is a research codebase for continual reinforcement learning, with Knowledge Evolution, Skill Evolution, SPT/SPI, and executable Modules as core concepts.
- `PROJECT_BIBLE.md` is the authoritative theory and terminology specification. Preserve its distinctions: Knowledge stores evidence-supported environment mechanisms; SPT is a skill-family generator; SPI is a task-specific learning specification; Policy is the controller; Module is the qualified executable skill.
- Before changing a research claim or experiment design, read the relevant current document under `docs/`. `README.md` points to current status and priorities. Status reports and candidate protocols are time-sensitive; do not treat a pilot, diagnostic, smoke test, or candidate manifest as a formal result or frozen design.
- Keep code, configs, analysis, and claims aligned. If implementation changes affect a protocol or recorded result, update the relevant dated report or manifest as part of the change.

## Code Changes

- Follow the existing Python structure: reusable framework code belongs under `src/`, experiment runners and analysis under `experiments/`, reusable settings under `configs/`, and regression tests under `tests/`.
- Keep environment-specific behavior in environment adapters and task-specific behavior in the corresponding skill/task modules. Prefer existing contracts and data structures over parallel abstractions.
- Preserve explicit reproducibility metadata: configuration, seeds, runtime/device details, budgets, and result provenance. Do not silently change defaults or overwrite historical experiment definitions; version a changed protocol/config when comparability would be affected.
- Keep the `PROJECT_BIBLE.md` boundaries intact. In particular, do not infer Knowledge mechanisms from policy/module performance, register an unqualified SPI as a Module, or let diagnostic component results imply full KECRL validation.
- Do not edit generated outputs, checkpoints, datasets, logs, or archived results unless the task explicitly requires it. Never delete or clean experiment artifacts as part of routine code work.
- Preserve unrelated worktree changes. Inspect `git status` before editing, and keep changes scoped to the request.

## Experiments and Claims

- Treat smoke tests as interface/connectivity checks, pilots as calibration/integration evidence, and diagnostics as mechanism investigations. Report each with its actual scope; only use “formal” when the protocol was frozen and the run satisfies its gates.
- Keep train, validation, and final evaluation roles distinct. Do not use final evaluation results to select checkpoints, tune settings, or alter the protocol.
- Use paired seeds/scenes and explicit episode/step budgets where the protocol requires them. Record actual optimizer settings and runtime costs rather than relying only on declared config values.
- When an experiment or analysis rejects malformed provenance, missing metadata, or protocol mismatches, retain that fail-closed behavior.

## Validation

- Run focused tests for the changed area first, then broaden to `pytest` when changes touch shared contracts, training behavior, or multiple modules.
- The project uses pytest (`tests/` is configured in `pyproject.toml`). A CPU smoke entry point is `python -m experiments.run --config configs/smoke_cpu.yaml`; it is an interface check, not a research result.
- Check the local runtime with `python scripts/check_environment.py`. GPU/Crafter-specific tests may require separately installed optional dependencies and a compatible device; state clearly when those checks cannot run.
- Do not launch long training or formal-budget experiments merely to validate a code change. Prefer deterministic, bounded tests or the smallest relevant smoke configuration.

## Working Efficiently

- Read this file, then inspect only the relevant implementation, tests, config, and current research note; avoid loading the full project bible unless the task changes theory or terminology.
- Use existing scripts and tests before inventing new commands or tooling. Keep documentation updates concise and link to the canonical detailed protocol/report instead of duplicating it.
- Before finishing, summarize changed files, verification performed, and any material test limitation.

## Storage on This Server

- The canonical project directory is `/data/zl202621/KECRL`. Run commands there. `/home/zl202621/project/KECRL` is a compatibility symlink, not a second checkout.
- Keep datasets in `datasets/`, checkpoints/models in `checkpoints/`, experiment artifacts and durable evidence in `results/`, and logs in `logs/` under the canonical project directory. The existing LIBERO installation and its datasets are in `third_party/LIBERO/`; use `KECRL_LIBERO_DATA_ROOT` for that dataset location.
- Use `scripts/with_storage.sh <command> [arguments...]`, or source `scripts/storage_env.sh` in the current shell. These configure persistent cache and temporary directories under `cache/` and `tmp/`. New login and interactive Bash shells also load this configuration.
- Store download/package caches under `cache/`, including pip, Conda packages, Hugging Face and Torch. The old `~/.cache` and `~/.conda/pkgs` paths are compatibility symlinks. Existing Conda environments remain at their original locations; do not relocate installed environments as part of routine cleanup.
- Verify `/data` is mounted and the destination is writable before creating large files. Do not fall back to the system SSD. Do not use `/tmp`, `/var/tmp`, or the old home directory as the sole location for new experiment outputs, checkpoints, datasets, or evidence. `tmp/` is scratch space; completed evidence belongs in `results/` or `logs/`.
- Historical frozen protocols/configurations, SHA bindings, and result files must keep their original contents. A compatibility symlink does not preserve resolved absolute paths: old `Path.resolve()`-based bindings may reject the new location. For future research execution, version the affected storage paths and bindings before payload access; never patch historical SHA values, remove dependencies, or disable validation. Missing pre-maintenance `/tmp` artifacts remain missing until genuinely restored.
- Preserve hidden files, `.git`, untracked work, file modes and existing directory structure during migration. Verify copies before removing the original copies. Migration byte reads/checksums are file-transfer verification, not research analysis or experimental results.
- Keep caches, third-party payloads, scratch files and raw experiment artifacts out of Git. This storage change does not authorize training, search, qualification, or access to sealed evaluation data; Policy500, stabilization5 and the existing loss remain unchanged.
