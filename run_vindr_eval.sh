#!/usr/bin/env bash
# CheXOne abnormality grounding on the official VinDr-CXR test split.
# Run inside the venv that already runs chexone (venv-qwen).
#
#   bash run_vindr_eval.sh check      # paths, csv/DICOM match, decoding; no GPU
#   bash run_vindr_eval.sh prompt     # print per-class prompts; no GPU
#   bash run_vindr_eval.sh probe      # 3 images, print raw answers
#   bash run_vindr_eval.sh n 100      # random 100 images, 4 GPU shards
#   bash run_vindr_eval.sh full       # all 949 images with boxes
#   bash run_vindr_eval.sh eval       # recompute metrics from cache; no GPU
#   bash run_vindr_eval.sh vis        # figures from cache -> <OUT>/vis/; no GPU
#
# Overrides: GPUS=0,1 NSHARD=2 VINDR_ROOT=/path CHEXONE_MODEL=/path
# Only the prompt changes between experiments (model code = README Quick Start):
#   PROMPT=instruct|reason|reason_dbl|suffix   (REASONING=1 is shorthand for PROMPT=reason)
#   RUNNER=official|ours  PIXELS=rec|default  MAX_NEW_TOKENS=1024
# Every setup writes its own dir: ./vindr_off_<prompt>[_px<pixels>][_mt<tokens>]  (RUNNER=ours: ./vindr_result[_reason|_<prompt>])
# vis: VIS_PICK=per-class|random|best|miss|nobox  VIS_N=2  VIS_CLASS="Cardiomegaly"  VIS_ALL=1 (all boxes)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

VINDR_ROOT="${VINDR_ROOT:-/mnt/data/jinghansun/data/Vin-DR}"
export CHEXONE_MODEL="${CHEXONE_MODEL:-/mnt/data/jinghansun/Huggingface/StanfordAIMI/CheXOne}"
GPUS="${GPUS:-0,1,2,3}"
NSHARD="${NSHARD:-4}"
REASONING="${REASONING:-0}"
LIMIT="${LIMIT:-0}"
LOG_DIR="./logs/vindr"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

RUNNER="${RUNNER:-official}"            # official = README Quick Start code | ours = old local_runner
PIXELS="${PIXELS:-rec}"                 # official runner: rec (max_pixels=512*512) | default (bare processor)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
PROMPT="${PROMPT:-}"                    # instruct | reason | reason_dbl | suffix ; empty = instruct, or reason if REASONING=1

if [[ -n "${PROMPT}" ]]; then PNAME="${PROMPT}"
elif [[ "${REASONING}" == "1" ]]; then PNAME="reason"
else PNAME="instruct"; fi

# Each setup gets its own result dir, so runs never overwrite each other.
if [[ "${RUNNER}" == "official" ]]; then
  DEFAULT_OUT="./vindr_off_${PNAME}"
  [[ "${PIXELS}" != "rec" ]] && DEFAULT_OUT="${DEFAULT_OUT}_px${PIXELS}"
  [[ "${MAX_NEW_TOKENS}" != "1024" ]] && DEFAULT_OUT="${DEFAULT_OUT}_mt${MAX_NEW_TOKENS}"
elif [[ "${PNAME}" == "instruct" ]]; then DEFAULT_OUT="./vindr_result"
elif [[ "${PNAME}" == "reason" ]]; then   DEFAULT_OUT="./vindr_result_reason"
else DEFAULT_OUT="./vindr_result_${PNAME}"; fi
OUT="${OUT:-${DEFAULT_OUT}}"            # an explicit OUT is used as given

# Flags that decide the cache key. Shared by every subcommand including vis, so they cannot drift.
RUN_FLAGS=(--runner "${RUNNER}" --pixels "${PIXELS}" --max-new-tokens "${MAX_NEW_TOKENS}")
if [[ -n "${PROMPT}" ]]; then RUN_FLAGS+=(--prompt "${PROMPT}")
elif [[ "${REASONING}" == "1" ]]; then RUN_FLAGS+=(--reasoning); fi

ARGS=(--data-root "${VINDR_ROOT}" --model "${CHEXONE_MODEL}" --out "${OUT}" "${RUN_FLAGS[@]}")

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
    CUDA_VISIBLE_DEVICES="${gpu_list[i]}" python -u vindr_chexone.py "${args[@]}" \
        --shard "${i}" --n-shard "${NSHARD}" --no-eval > "${base}_s${i}.log" 2>&1 &
    pids+=($!)
  done
  echo "[vindr] ${NSHARD} shards started. Progress: tail -f ${base}_s0.log"
  for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
      echo "[vindr] shard ${i} FAILED, see ${base}_s${i}.log"
      failed=1
    fi
  done
  [[ "${failed}" -eq 0 ]] || return 1
  python -u vindr_chexone.py "${args[@]}" --reparse-only 2>&1 | tee "${base}_eval.log"
}

CMD="${1:-help}"
case "${CMD}" in
  check)  python -u vindr_chexone.py "${ARGS[@]}" --check ;;
  prompt) python -u vindr_chexone.py "${ARGS[@]}" --dry-run ;;
  probe)
    IFS=',' read -ra first_gpu <<< "${GPUS}"
    CUDA_VISIBLE_DEVICES="${first_gpu[0]}" python -u vindr_chexone.py "${ARGS[@]}" \
        --limit 3 --show-raw 8 2>&1 | tee "${LOG_DIR}/probe_${STAMP}.log"
    ;;
  n)
    LIMIT="${2:-}"
    if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]] || [[ "${LIMIT}" -eq 0 ]]; then
      echo "usage: bash run_vindr_eval.sh n <images>"
      exit 1
    fi
    run_shards
    ;;
  full) run_shards ;;
  eval)
    python -u vindr_chexone.py "${ARGS[@]}" --limit "${LIMIT}" --reparse-only \
        2>&1 | tee "${LOG_DIR}/eval_${STAMP}.log"
    ;;
  vis)
    VARGS=(--data-root "${VINDR_ROOT}" --out "${OUT}" "${RUN_FLAGS[@]}"
           --pick "${VIS_PICK:-per-class}" --n "${VIS_N:-2}")
    [[ -n "${VIS_CLASS:-}" ]] && VARGS+=(--class "${VIS_CLASS}")
    [[ "${VIS_ALL:-0}" == "1" ]] && VARGS+=(--all-boxes)
    python -u vindr_vis.py "${VARGS[@]}" 2>&1 | tee "${LOG_DIR}/vis_${STAMP}.log"
    ;;
  *) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
