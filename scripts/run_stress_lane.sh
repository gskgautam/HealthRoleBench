#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo 'Usage: run_stress_lane.sh PROJECT_ROOT GPU' >&2
    exit 2
fi
root=$(cd -- "$1" && pwd)
gpu=$2
repo=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ "$gpu" != 0 && "$gpu" != 1 ]]; then
    echo 'Select GPU 0 or 1' >&2
    exit 2
fi
cd -- "$root"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9>"$root/logs/encoder_transfer_v1/gpu$gpu.lock"
echo "Waiting for GPU $gpu experiment lane"
flock -w 3600 9
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if grep -Fxq "$uuid" <<< "$apps"; then
    echo "GPU $gpu has an active compute process" >&2
    exit 1
fi
for model in llama2 mistral qwen gemma; do
    target="data/experiments/zero_shot_cue_mask_v1/$model.csv"
    [[ ! -f "$target" ]] || continue
    python3 -u -m healthrolebench.models.zero_shot "$model" --condition cue-masked
done
summary=data/experiments/zero_shot_cue_mask_v1/table_18_cue_masking_pilot.csv
[[ -f "$summary" ]] || python3 -m healthrolebench.reporting.stress_tables --project-root "$root"
