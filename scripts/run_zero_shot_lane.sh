#!/usr/bin/env bash
# Zero-shot classification (Tables 9, 10, 13, 18; Figure 8).
# Usage: [ITEMS=eval] [CONDITIONS="original cue-masked"] run_zero_shot_lane.sh PROJECT_ROOT GPU MODEL [MODEL...]
# ITEMS=audit (450-post audit sample, default) or eval (1,500-post evaluation set).
# CONDITIONS: any of original, cue-masked, paraphrased, adversarial (rewrites: eval only).
# Completed runs are skipped.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'Usage: run_zero_shot_lane.sh PROJECT_ROOT GPU MODEL [MODEL...]' >&2
    exit 2
fi
root=$(cd -- "$1" && pwd)
gpu=$2
shift 2
repo=$(cd -- "$(dirname -- "$0")" && pwd)
cd -- "$root"
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
mkdir -p logs/encoder_transfer_v1
exec 9>"logs/encoder_transfer_v1/gpu$gpu.lock"   # shared with the other GPU lanes
flock -n 9 || { echo "Another experiment lane holds GPU $gpu" >&2; exit 1; }
items=${ITEMS:-audit}
for model in "$@"; do
    for condition in ${CONDITIONS:-original cue-masked}; do
        if [[ $items == eval ]]; then dir=zero_shot_eval_v1/$condition
        else dir=$([[ $condition == original ]] && echo zero_shot_audit_v2 || echo zero_shot_cue_mask_v2); fi
        [[ ! -f "data/experiments/$dir/$model.csv" ]] || { echo "skip $model $items $condition"; continue; }
        echo "$(date -Is) GPU $gpu $model $items $condition"
        python3 -u -m healthrolebench.models.zero_shot "$model" --items "$items" --condition "$condition"
    done
done
echo "$(date -Is) lane complete: $*"
