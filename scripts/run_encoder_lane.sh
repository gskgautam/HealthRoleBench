#!/usr/bin/env bash
set -euo pipefail
# PROTOCOL=held_out (default; train on two diseases, test on the third: Tables 7, 12-14)
# PROTOCOL=source   (train on one disease, test on each other disease: Table 11)
if [[ $# -lt 3 ]]; then
    echo 'Usage: [PROTOCOL=held_out|source] run_encoder_lane.sh PROJECT_ROOT GPU MODEL [MODEL...]' >&2
    exit 2
fi
root=$(cd -- "$1" && pwd)
gpu=$2
shift 2
if [[ "$gpu" != 0 && "$gpu" != 1 ]]; then
    echo 'Select GPU 0 or 1.' >&2
    exit 2
fi
repo=$(cd -- "$(dirname -- "$0")" && pwd)
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" HF_HUB_DISABLE_TELEMETRY=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$root/logs/encoder_transfer_v1"
exec 9>"$root/logs/encoder_transfer_v1/gpu$gpu.lock"
flock -n 9 || { echo "This project already has an encoder queue on GPU $gpu" >&2; exit 1; }
for seed in 42 43 44; do
    for model in "$@"; do
        for disease in COVID-19 HPV Influenza; do
            if [[ ${PROTOCOL:-held_out} == source ]]; then folder=encoder_source_v1 flag=--source-disease
            else folder=encoder_transfer_v1 flag=--disease; fi
            result="$root/data/experiments/$folder/$model/$disease/seed_$seed/metrics.json"
            [[ ! -f "$result" ]] || continue
            uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
            apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
            if grep -Fxq "$uuid" <<< "$apps"; then
                echo "GPU $gpu has an active compute process; pausing lane" >&2
                exit 1
            fi
            printf 'GPU %s %s %s seed %s\n' "$gpu" "$model" "$disease" "$seed"
            log=$(mktemp "$root/logs/encoder_transfer_v1/$model.$disease.$seed.XXXXXX")
            if ! python3 -u -m healthrolebench.models.train_encoder --project-root "$root" --model "$model" \
                 "$flag" "$disease" --seed "$seed" >"$log" 2>&1; then
                echo "Training failed: $log" >&2
                tail -15 "$log" >&2
                exit 1
            fi
            tail -1 "$log"
        done
    done
done
