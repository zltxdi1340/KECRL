#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    printf 'Usage: %s <command> [arguments...]\n' "$0" >&2
    exit 2
fi

source /data/zl202621/KECRL/scripts/storage_env.sh
cd "$KECRL_ROOT"
exec "$@"
