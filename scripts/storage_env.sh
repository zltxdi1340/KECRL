# Source this file to configure this server's persistent KECRL storage.
# It intentionally leaves device, seeds, optimizer settings and budgets alone.

if ! mountpoint -q /data || [ ! -d /data/zl202621/KECRL ] || [ ! -w /data/zl202621/KECRL ]; then
    printf '%s\n' 'KECRL storage is unavailable: /data/zl202621/KECRL must be mounted and writable.' >&2
    return 1
fi

export KECRL_ROOT=/data/zl202621/KECRL
export KECRL_DATA_ROOT="$KECRL_ROOT/datasets"
export KECRL_RESULTS_ROOT="$KECRL_ROOT/results"
export KECRL_CHECKPOINT_ROOT="$KECRL_ROOT/checkpoints"
export KECRL_LIBERO_ROOT="$KECRL_ROOT/third_party/LIBERO"
export KECRL_LIBERO_DATA_ROOT="$KECRL_LIBERO_ROOT/datasets"
export KECRL_CACHE_ROOT="$KECRL_ROOT/cache"
export KECRL_TMP_ROOT="$KECRL_ROOT/tmp"

export XDG_CACHE_HOME="$KECRL_CACHE_ROOT"
export PIP_CACHE_DIR="$KECRL_CACHE_ROOT/pip"
export CONDA_PKGS_DIRS="$KECRL_CACHE_ROOT/conda/pkgs"
export HF_HOME="$KECRL_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$KECRL_CACHE_ROOT/torch"
export UV_CACHE_DIR="$KECRL_CACHE_ROOT/uv"
export MPLCONFIGDIR="$KECRL_CACHE_ROOT/matplotlib"
export NUMBA_CACHE_DIR="$KECRL_CACHE_ROOT/numba"
export TRITON_CACHE_DIR="$KECRL_CACHE_ROOT/triton"
export WANDB_CACHE_DIR="$KECRL_CACHE_ROOT/wandb"
export WANDB_DIR="$KECRL_ROOT/logs/wandb"
export TMPDIR="$KECRL_TMP_ROOT"
export TEMP="$KECRL_TMP_ROOT"
export TMP="$KECRL_TMP_ROOT"

mkdir -p -- "$KECRL_DATA_ROOT" "$KECRL_RESULTS_ROOT" "$KECRL_CHECKPOINT_ROOT" \
    "$KECRL_CACHE_ROOT" "$KECRL_TMP_ROOT" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
    "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$UV_CACHE_DIR" \
    "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$WANDB_CACHE_DIR" "$WANDB_DIR"
