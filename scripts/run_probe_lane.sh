#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || ( "$2" != 0 && "$2" != 1 ) ]]; then
    echo 'Usage: run_probe_lane.sh PROJECT_ROOT GPU' >&2
    exit 2
fi
root=$(cd -- "$1" && pwd)
gpu=$2
repo=$(cd -- "$(dirname -- "$0")" && pwd)
cd -- "$root"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec 9>"$root/logs/encoder_transfer_v1/gpu$gpu.lock"
echo "Waiting for GPU $gpu experiment lane"
flock -w 7200 9
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fxq "$uuid" <<< "$apps"; then
    echo "GPU $gpu has an active compute process" >&2
    exit 1
fi
for model in llama2 mistral qwen gemma; do
    [[ ! -f "data/experiments/frozen_head_audit_v1/$model.csv" ]] || continue
    python3 -u -m healthrolebench.models.probe --project-root "$root" --model "$model"
done
summary=data/experiments/frozen_head_audit_v1/table_10_zero_shot_vs_frozen_head_pilot.csv
[[ -f "$summary" ]] || python3 -m healthrolebench.reporting.probe_tables --project-root "$root"
