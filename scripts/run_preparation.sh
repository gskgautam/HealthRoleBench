#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo 'Usage: run_preparation.sh PROJECT_ROOT' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$repo" OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
cd -- "$1"
python3 -m healthrolebench.curation.corpus_tables
python3 -m healthrolebench.curation.exemplar_links
python3 -m healthrolebench.analysis.full_transfer
