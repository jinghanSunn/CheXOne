#!/usr/bin/env bash
# CheXOne on the paper's Long-tail Disease Classification (CXRLongtail: 5 diseases x 150, binary, accuracy).
# Run inside the venv that already runs chexone (venv-qwen).
#
#   bash run_cxrlt_eval.sh check     # csv columns, sample counts, image files exist; no GPU
#   bash run_cxrlt_eval.sh prompt    # print sampling stats + 3 example prompts; no GPU
#   bash run_cxrlt_eval.sh probe     # 5 samples per disease on 1 GPU, print raw answers
#   bash run_cxrlt_eval.sh n 30      # 30 per disease (150 total), 4 GPU shards
#   bash run_cxrlt_eval.sh full      # 150 per disease (750 total), 4 GPU shards
#   bash run_cxrlt_eval.sh eval      # recompute accuracy from cache; no GPU
#
# Required: CXRLT_CSV=.../miccai-2023_mimic-cxr-lt/miccai2023_mimic-cxr-lt_labels_test.csv
#           MIMIC_JPG=.../mimic-cxr-jpg/2.0.0/files
# Unknowns of the paper are knobs (each setting writes its own OUT/cache entries):
#   PROMPT=mc_reason|mc_reason_paren|mc|mc_reason_dbl|yn_reason  POS_FRAC=0.5  VIEWS=study|one
#   NEG_POOL=random|nofinding  SHUFFLE=1   (paper eval script: MAX_NEW_TOKENS=2048)
# Also: GPUS=0,1,2,3 NSHARD=4 PIXELS=rec|default MAX_NEW_TOKENS=1024 CHEXONE_MODEL=/path
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

CXRLT_ROOT="${CXRLT_ROOT:-/mnt/data/jinghansun/data/cxr-lt-multi-label-long-tailed-classification-on-chest-x-rays-2.0.0}"
CXRLT_CSV="${CXRLT_CSV:-${CXRLT_ROOT}/miccai-2023_mimic-cxr-lt/miccai2023_mimic-cxr-lt_labels_test.csv}"
MIMIC_JPG="${MIMIC_JPG:-/mnt/data/jinghansun/data/MIMIC-CXR/2.0.0/files}"
export CHEXONE_MODEL="${CHEXONE_MODEL:-/mnt/data/jinghansun/Huggingface/StanfordAIMI/CheXOne}"
GPUS="${GPUS:-0,1,2,3}"
NSHARD="${NSHARD:-4}"
PROMPT="${PROMPT:-mc_reason}"
POS_FRAC="${POS_FRAC:-0.5}"
VIEWS="${VIEWS:-study}"
NEG_POOL="${NEG_POOL:-random}"         # random | nofinding
SHUFFLE="${SHUFFLE:-1}"
PIXELS="${PIXELS:-rec}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LIMIT="${LIMIT:-0}"
LOG_DIR="./logs/cxrlt"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

DEFAULT_OUT="./cxrlt_${PROMPT}_pos${POS_FRAC}_${VIEWS}"
[[ "${NEG_POOL}" != "random" ]] && DEFAULT_OUT="${DEFAULT_OUT}_neg${NEG_POOL}"
[[ "${SHUFFLE}" != "1" ]] && DEFAULT_OUT="${DEFAULT_OUT}_noshuf"
[[ "${PIXELS}" != "rec" ]] && DEFAULT_OUT="${DEFAULT_OUT}_px${PIXELS}"
[[ "${MAX_NEW_TOKENS}" != "1024" ]] && DEFAULT_OUT="${DEFAULT_OUT}_mt${MAX_NEW_TOKENS}"
OUT="${OUT:-${DEFAULT_OUT}}"

ARGS=(--labels-csv "${CXRLT_CSV}" --jpg-root "${MIMIC_JPG}" --model "${CHEXONE_MODEL}"
      --out "${OUT}" --prompt "${PROMPT}" --pos-frac "${POS_FRAC}" --views "${VIEWS}"
      --neg-pool "${NEG_POOL}"
      --pixels "${PIXELS}" --max-new-tokens "${MAX_NEW_TOKENS}")
[[ "${SHUFFLE}" != "1" ]] && ARGS+=(--no-shuffle-options)

if [[ ! -f "${CXRLT_CSV}" && "${1:-help}" != "help" ]]; then
  echo "[ERROR] CXRLT_CSV not found: ${CXRLT_CSV}"
  echo "  download from PhysioNet cxr-lt-iccv-workshop-cvamd v2.0.0:"
  echo "  miccai-2023_mimic-cxr-lt/miccai2023_mimic-cxr-lt_labels_test.csv"
  exit 1
fi

run_shards() {
  local args=("${ARGS[@]}" --limit "${LIMIT}")
  local base="${LOG_DIR}/full_${STAMP}"
  local gpu_list pids=() i failed=0
  IFS=',' read -ra gpu_list <<< "${GPUS}"
  if [[ "${#gpu_list[@]}" -lt "${NSHARD}" ]]; then
    echo "[ERROR] NSHARD=${NSHARD} but GPUS has only ${#gpu_list[@]} entries"
    return 1
  fi
  for ((i = 0; i < NSHARD; i++)); do
    CUDA_VISIBLE_DEVICES="${gpu_list[i]}" python -u cxrlt_chexone.py "${args[@]}" \
        --shard "${i}" --n-shard "${NSHARD}" --no-eval > "${base}_s${i}.log" 2>&1 &
    pids+=($!)
  done
  echo "[cxrlt] ${NSHARD} shards started. Progress: tail -f ${base}_s0.log"
  for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
      echo "[cxrlt] shard ${i} FAILED, see ${base}_s${i}.log"
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]] || return 1
  python -u cxrlt_chexone.py "${args[@]}" --reparse-only 2>&1 | tee "${base}_eval.log"
}

CMD="${1:-help}"
case "${CMD}" in
  check)  python -u cxrlt_chexone.py "${ARGS[@]}" --check ;;
  prompt) python -u cxrlt_chexone.py "${ARGS[@]}" --dry-run ;;
  probe)
    IFS=',' read -ra first_gpu <<< "${GPUS}"
    CUDA_VISIBLE_DEVICES="${first_gpu[0]}" python -u cxrlt_chexone.py "${ARGS[@]}" \
        --limit 5 --show-raw 6 2>&1 | tee "${LOG_DIR}/probe_${STAMP}.log"
    ;;
  n)
    LIMIT="${2:-}"
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || [[ "${LIMIT}" -eq 0 ]]; then
      echo "usage: bash run_cxrlt_eval.sh n <per-disease>"
      exit 1
    fi
    run_shards
    ;;
  full) run_shards ;;
  eval)
    python -u cxrlt_chexone.py "${ARGS[@]}" --limit "${LIMIT}" --reparse-only \
        2>&1 | tee "${LOG_DIR}/eval_${STAMP}.log"
    ;;
  *) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
