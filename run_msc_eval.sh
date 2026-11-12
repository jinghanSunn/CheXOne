#!/usr/bin/env bash
# CheXOne phrase grounding on MS-CXR (paper: mIoU 0.608, mAP 0.835, 127 samples).
# Run inside the venv that already runs chexone (venv-qwen).
#
#   bash run_msc_eval.sh check      # annotations, MIMIC jpg present, sizes match; no GPU
#   bash run_msc_eval.sh prompt     # print the first prompts; no GPU
#   bash run_msc_eval.sh probe      # 3 images, print raw answers
#   bash run_msc_eval.sh n 50       # random 50 images, 4 GPU shards
#   bash run_msc_eval.sh full       # whole split (test: 155 images, 176 queries)
#   bash run_msc_eval.sh eval       # recompute metrics from cache; no GPU
#
# Data:   MSCXR_ANN=<csv | release zip | dir>   MIMIC_ROOT=<MIMIC-CXR-JPG 2.0.0 dir, has files/p10/...>
# Choose: SPLIT=test|val|train|all   GPUS=0,1 NSHARD=2 CHEXONE_MODEL=/path
# Prompt: PROMPT=instruct|reason|reason_dbl|suffix  (REASONING=1 = reason)
#         RUNNER=official|ours  PIXELS=rec|default  MAX_NEW_TOKENS=1024
# Every setup writes its own dir: ./msc_<split>_<prompt>[_<runner>][_px<pixels>][_mt<tokens>]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

MSCXR_ANN="${MSCXR_ANN:-/mnt/data/jinghansun/data/MS-CXR/ms-cxr-making-the-most-of-text-semantics-to-improve-biomedical-vision-language-processing-1.1.0}"
MIMIC_ROOT="${MIMIC_ROOT:-/mnt/data/jinghansun/data/MIMIC-CXR/2.0.0}"
export CHEXONE_MODEL="${CHEXONE_MODEL:-/mnt/data/jinghansun/Huggingface/StanfordAIMI/CheXOne}"
GPUS="${GPUS:-0,1,2,3}"
NSHARD="${NSHARD:-4}"
REASONING="${REASONING:-0}"
LIMIT="${LIMIT:-0}"
SPLIT="${SPLIT:-test}"
LOG_DIR="./logs/msc"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

RUNNER="${RUNNER:-official}"
PIXELS="${PIXELS:-rec}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PROMPT="${PROMPT:-}"

if [[ -n "${PROMPT}" ]]; then PNAME="${PROMPT}"
elif [[ "${REASONING}" == "1" ]]; then PNAME="reason"
else PNAME="instruct"; fi

DEFAULT_OUT="./msc_${SPLIT}_${PNAME}"
[[ "${RUNNER}" != "official" ]] && DEFAULT_OUT="${DEFAULT_OUT}_${RUNNER}"
[[ "${PIXELS}" != "rec" ]] && DEFAULT_OUT="${DEFAULT_OUT}_px${PIXELS}"
[[ "${MAX_NEW_TOKENS}" != "1024" ]] && DEFAULT_OUT="${DEFAULT_OUT}_mt${MAX_NEW_TOKENS}"
OUT="${OUT:-${DEFAULT_OUT}}"

RUN_FLAGS=(--runner "${RUNNER}" --pixels "${PIXELS}" --max-new-tokens "${MAX_NEW_TOKENS}")
if [[ -n "${PROMPT}" ]]; then RUN_FLAGS+=(--prompt "${PROMPT}")
elif [[ "${REASONING}" == "1" ]]; then RUN_FLAGS+=(--reasoning); fi

ARGS=(--ann "${MSCXR_ANN}" --jpg-root "${MIMIC_ROOT}" --split "${SPLIT}"
      --model "${CHEXONE_MODEL}" --out "${OUT}" "${RUN_FLAGS[@]}")

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
    CUDA_VISIBLE_DEVICES="${gpu_list[i]}" python -u ms_cxr_chexone.py "${args[@]}" \
        --shard "${i}" --n-shard "${NSHARD}" --no-eval > "${base}_s${i}.log" 2>&1 &
    pids+=($!)
  done
  echo "[msc] ${NSHARD} shards started. Progress: tail -f ${base}_s0.log"
  for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
      echo "[msc] shard ${i} FAILED, see ${base}_s${i}.log"
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]] || return 1
  python -u ms_cxr_chexone.py "${args[@]}" --reparse-only 2>&1 | tee "${base}_eval.log"
}

CMD="${1:-help}"
case "${CMD}" in
  check)  python -u ms_cxr_chexone.py "${ARGS[@]}" --check ;;
  prompt) python -u ms_cxr_chexone.py "${ARGS[@]}" --dry-run ;;
  probe)
    IFS=',' read -ra first_gpu <<< "${GPUS}"
    CUDA_VISIBLE_DEVICES="${first_gpu[0]}" python -u ms_cxr_chexone.py "${ARGS[@]}" \
        --limit 3 --show-raw 8 2>&1 | tee "${LOG_DIR}/probe_${STAMP}.log"
    ;;
  n)
    LIMIT="${2:-}"
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || [[ "${LIMIT}" -eq 0 ]]; then
      echo "usage: bash run_msc_eval.sh n <images>"
      exit 1
    fi
    run_shards
    ;;
  full) run_shards ;;
  eval)
    python -u ms_cxr_chexone.py "${ARGS[@]}" --limit "${LIMIT}" --reparse-only \
        2>&1 | tee "${LOG_DIR}/eval_${STAMP}.log"
    ;;
  *) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
