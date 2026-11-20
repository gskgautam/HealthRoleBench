#!/usr/bin/env bash
# Full-data frozen-backbone classifiers for Tables 8 and 10.
# Usage: run_probe_full_lane.sh PROJECT_ROOT GPU [MODEL...]   (default: qwen llama2 mistral gemma)
# Embeds all posts once per model (skipped if cached), then trains and scores the heads.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo 'Usage: run_probe_full_lane.sh PROJECT_ROOT GPU [MODEL...]' >&2
    exit 2
fi
root=$(cd -- "$1" && pwd)
gpu=$2
shift 2
models=("${@:-qwen llama2 mistral gemma}")
repo=$(cd -- "$(dirname -- "$0")" && pwd)
cd -- "$root"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
mkdir -p logs/encoder_transfer_v1
exec 9>"logs/encoder_transfer_v1/gpu$gpu.lock"   # shared with the other GPU lanes
flock -n 9 || { echo "Another experiment lane holds GPU $gpu" >&2; exit 1; }
for model in ${models[*]}; do
    [[ ! -f "data/experiments/probe_full_v1/embeddings/$model.npy" ]] || { echo "skip embed $model"; continue; }
    echo "$(date -Is) GPU $gpu embed $model"
    python3 -u -m healthrolebench.models.probe_full embed --project-root . --model "$model"
done
[[ -f data/experiments/probe_full_v1/table_08_full.csv ]] || python3 -u -m healthrolebench.models.probe_full heads --project-root .
echo "$(date -Is) probe lane complete"
