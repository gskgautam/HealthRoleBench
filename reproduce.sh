#!/usr/bin/env bash
# Reproduce HealthRoleBench Tables 1-20 and Figures 5-8 in dependency order.
#
# Usage:  ./reproduce.sh PROJECT_ROOT [STAGE...]        (default: all stages, in order)
# Stages: corpus groups thresholds evalset encoders rewrites zeroshot gpt4o probes domain tables index
# Env:    GPU_A / GPU_B  GPU ids for the two lanes (default 0 and 1)
#         OPENAI_API_KEY needed only for the gpt4o stage (about US$3.30 for four conditions)
#         HRB_HF_CACHE, HRB_GEMMA_CACHE  model caches if not in the default locations
# Every step writes its outputs once and refuses to overwrite; completed lane runs are skipped,
# so the script can be re-run after an interruption. See README section 3 for each step.
set -euo pipefail
if [[ $# -lt 1 ]]; then sed -n '2,12p' "$0"; exit 2; fi
root=$(cd -- "$1" && pwd); shift
repo=$(cd -- "$(dirname -- "$0")" && pwd)
stages=("${@:-corpus groups thresholds evalset encoders rewrites zeroshot gpt4o probes domain tables index}")
a=${GPU_A:-0}; b=${GPU_B:-1}
cd -- "$root"
export PYTHONPATH="$repo" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
E=data/experiments
py() { echo "+ python -m healthrolebench.$*"; python3 -u -m "healthrolebench.$@"; }
once() { [[ -e "$1" ]] && echo "skip ($1 exists)" || { shift; "$@"; }; }

for stage in ${stages[*]}; do
  echo "=== $stage ($(date -Is))"
  case $stage in
    corpus)     once $E/corpus_tables_v1 py corpus_tables
                once $E/definition_tables_v1 py definition_tables --project-root . ;;
    groups)     once $E/corpus_tables_v1/topup_exemplar_links.csv py exemplar_links
                once $E/full_transfer_pilot_v1 py full_transfer ;;
    thresholds) once $E/threshold_f1_v1 py threshold_f1 --project-root . ;;
    evalset)    once $E/eval_set_v1 py eval_set --project-root . ;;
    encoders)   "$repo/scripts/run_encoder_lane.sh" . "$a" bert roberta distilbert
                PROTOCOL=source "$repo/scripts/run_encoder_lane.sh" . "$a" bert roberta distilbert ;;
    rewrites)   for c in paraphrased adversarial; do
                  once $E/eval_set_v1/rewrites_$c.csv env CUDA_VISIBLE_DEVICES="$a" python3 -u -m healthrolebench.analysis.perturb generate --condition $c
                  once $E/eval_set_v1/perturbed_$c.csv env CUDA_VISIBLE_DEVICES="$a" python3 -u -m healthrolebench.analysis.perturb validate --condition $c
                done ;;
    zeroshot)   ITEMS=eval CONDITIONS="original cue-masked paraphrased adversarial" "$repo/scripts/run_zero_shot_lane.sh" . "$a" qwen llama2 mistral gemma ;;
    gpt4o)      [[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'gpt4o stage needs OPENAI_API_KEY' >&2; exit 1; }
                for c in original cue-masked paraphrased adversarial; do
                  once $E/zero_shot_eval_v1/$c/gpt4o.csv py zero_shot_api --items eval --condition $c --max-usd 1.00
                done ;;
    probes)     "$repo/scripts/run_probe_full_lane.sh" . "$b" ;;
    domain)     CUDA_VISIBLE_DEVICES="$b" py domain_adapt all --project-root . ;;
    tables)     once $E/encoder_transfer_v1/table_12_encoder_macro_f1.csv py report_tables encoders --project-root .
                once $E/full_transfer_pilot_v1/table_s1_fold_training_coverage.csv py report_tables coverage --project-root .
                once $E/supplementary_v1/table_s2_prompt_echo.csv py report_tables echo --project-root .
                py encoder_tables --project-root .
                once $E/eval_tables_v1 py eval_tables --project-root .
                once $E/significance_v1 py significance --project-root . ;;
    index)      py paper_index --project-root . --repo-root "$repo" ;;
    *)          echo "Unknown stage: $stage" >&2; exit 2 ;;
  esac
done
echo "=== done ($(date -Is))"
