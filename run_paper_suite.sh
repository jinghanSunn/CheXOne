#!/usr/bin/env bash
# Re-run the CheXOne paper benchmarks we care about (arXiv 2604.00493, §4.3), one setting for all:
# official Quick Start runner, Reasoning mode (paper default), MAX_PIXELS=512*512, max_new_tokens=2048
# (paper eval script). Run inside venv-qwen.
#
#   part          data                                         paper
#   view          MIMIC-CXR test, 300 AP/PA/lateral            acc 0.9978 (0.978-1.0), n=900
#   longtail      CXR-LT (MIMIC-CXR-LT) test, 5 x 150          acc 0.816 (0.791-0.842), n=750
#   phrase        MS-CXR test, (image, phrase)                 mIoU 0.608 / mAP 0.835, n=127
#   abnormality   VinDr-CXR test, (image, class)               mIoU 0.452 / mAP 0.471, n=743
#   findings      MIMIC-CXR test, ReXRank json, 2347 studies   ReXRank CheXOne-R1: 1/RadCliQ 1.023 ...
#
#   bash run_paper_suite.sh check            # every part: data present, no GPU
#   bash run_paper_suite.sh probe            # every part: a few samples, raw answers printed
#   bash run_paper_suite.sh run [parts...]   # full run (default: all parts, one after another)
#   bash run_paper_suite.sh metrics          # findings: BLEU-2 / ROUGE-L / BertScore, local model only
#   bash run_paper_suite.sh summary          # one table: ours vs paper, no GPU
#   bash run_paper_suite.sh vis [parts...]   # per-sample figures: image(s) + GT + prediction, no GPU
#        VIS_PICK=mix|wrong|right|random  VIS_N=12  VIS_FILTER=Pneumoperitoneum  -> <SUITE_DIR>/<part>/vis/
#
# Overrides: GPUS=0,1,2,3 NSHARD=4 SUITE_DIR=./paper_suite MAX_NEW_TOKENS=2048 CHEXONE_MODEL=/path
#            SMALL=1  (run: 50 images/class or studies per part, to check the pipeline first)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

export CHEXONE_MODEL="${CHEXONE_MODEL:-/mnt/data/jinghansun/Huggingface/StanfordAIMI/CheXOne}"
MIMIC_ROOT="${MIMIC_ROOT:-/mnt/data/jinghansun/data/MIMIC-CXR/2.0.0}"
export GPUS="${GPUS:-0,1,2,3}"
export NSHARD="${NSHARD:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export PIXELS="${PIXELS:-rec}"
SUITE_DIR="${SUITE_DIR:-./paper_suite}"
BERT_MODEL="${BERT_MODEL:-/mnt/data/jinghansun/Huggingface/Bert/distilbert/distilbert-base-uncased}"  # BertScore
SMALL="${SMALL:-0}"
LOG_DIR="./logs/suite"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}" "${SUITE_DIR}"
ALL_PARTS=(view longtail phrase abnormality findings)

COMMON=(--model "${CHEXONE_MODEL}" --pixels "${PIXELS}" --max-new-tokens "${MAX_NEW_TOKENS}")
# MIMIC csvs (metadata / split): view_chexone.py looks in MIMIC_ROOT and its parent
# (server: /mnt/data/jinghansun/data/MIMIC-CXR/), .csv or .csv.gz. Force a dir with MIMIC_CSV_DIR=...
VIEW_ARGS=(--mimic-root "${MIMIC_ROOT}" --out "${SUITE_DIR}/view" --cache-dir ./cache_view "${COMMON[@]}")
if [[ -n "${MIMIC_CSV_DIR:-}" ]]; then
  VIEW_ARGS+=(--meta-csv "${MIMIC_CSV_DIR}/mimic-cxr-2.0.0-metadata.csv"
              --split-csv "${MIMIC_CSV_DIR}/mimic-cxr-2.0.0-split.csv")
fi
# findings test set: MIMIC annotation.json (R2Gen format, test = same 2347 studies as ReXRank).
# data/ReXRank_MIMICCXR_test.json also works: FIND_JSON=./data/ReXRank_MIMICCXR_test.json
FIND_JSON="${FIND_JSON:-$(dirname "${MIMIC_ROOT}")/annotation.json}"
FIND_ARGS=(--test-json "${FIND_JSON}" --jpg-root "${MIMIC_ROOT}/files" --out "${SUITE_DIR}/findings"
           --cache-dir ./cache_findings "${COMMON[@]}")

# generic: shard a python script over GPUS, then re-run on cache to score
shard_py() {   # $1 = script, rest = args
  local script="$1"; shift
  local args=("$@") base="${LOG_DIR}/${script%.py}_${STAMP}" gpu_list pids=() i failed=0
  IFS=',' read -ra gpu_list <<< "${GPUS}"
  if [[ "${#gpu_list[@]}" -lt "${NSHARD}" ]]; then
    echo "[ERROR] NSHARD=${NSHARD} but GPUS has only ${#gpu_list[@]} entries"; return 1
  fi
  for ((i = 0; i < NSHARD; i++)); do
    CUDA_VISIBLE_DEVICES="${gpu_list[i]}" python -u "${script}" "${args[@]}" \
        --shard "${i}" --n-shard "${NSHARD}" --no-eval > "${base}_s${i}.log" 2>&1 &
    pids+=($!)
  done
  echo "[suite] ${script}: ${NSHARD} shards. Progress: tail -f ${base}_s0.log"
  for i in "${!pids[@]}"; do
    wait "${pids[i]}" || { echo "[suite] shard ${i} FAILED, see ${base}_s${i}.log"; failed=1; }
  done
  [[ "${failed}" -eq 0 ]] || return 1
  python -u "${script}" "${args[@]}" --reparse-only 2>&1 | tee "${base}_eval.log"
}

# part -> command. mode: check | probe | run
part() {
  local name="$1" mode="$2" lim=()
  case "${name}" in
    view)
      [[ "${SMALL}" == "1" ]] && lim=(--limit 50)
      case "${mode}" in
        check) python -u view_chexone.py "${VIEW_ARGS[@]}" --check ;;
        probe) CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -u view_chexone.py "${VIEW_ARGS[@]}" --limit 2 --show-raw 3 ;;
        run)   shard_py view_chexone.py "${VIEW_ARGS[@]}" ${lim[@]+"${lim[@]}"} ;;
      esac ;;
    longtail)
      local sub=full; [[ "${SMALL}" == "1" ]] && sub="n 50"
      [[ "${mode}" == "check" ]] && sub=check; [[ "${mode}" == "probe" ]] && sub=probe
      OUT="${SUITE_DIR}/longtail" PROMPT=mc_reason bash run_cxrlt_eval.sh ${sub} ;;
    phrase)
      local sub=full; [[ "${SMALL}" == "1" ]] && sub="n 50"
      [[ "${mode}" == "check" ]] && sub=check; [[ "${mode}" == "probe" ]] && sub=probe
      OUT="${SUITE_DIR}/phrase" PROMPT=reason RUNNER=official bash run_msc_eval.sh ${sub} ;;
    abnormality)
      local sub=full; [[ "${SMALL}" == "1" ]] && sub="n 50"
      [[ "${mode}" == "check" ]] && sub=check; [[ "${mode}" == "probe" ]] && sub=probe
      OUT="${SUITE_DIR}/abnormality" PROMPT=reason RUNNER=official bash run_vindr_eval.sh ${sub} ;;
    findings)
      [[ "${SMALL}" == "1" ]] && lim=(--limit 50)
      case "${mode}" in
        check) python -u findings_chexone.py "${FIND_ARGS[@]}" --check ;;
        probe) CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -u findings_chexone.py "${FIND_ARGS[@]}" --limit 2 --show-raw 2 ;;
        run)   shard_py findings_chexone.py "${FIND_ARGS[@]}" ${lim[@]+"${lim[@]}"} ;;
      esac ;;
    *) echo "[suite] unknown part: ${name} (parts: ${ALL_PARTS[*]})"; return 1 ;;
  esac
}

CMD="${1:-help}"; shift || true
PARTS=("$@"); [[ "${#PARTS[@]}" -eq 0 ]] && PARTS=("${ALL_PARTS[@]}")
case "${CMD}" in
  check|probe|run)
    rc=0
    for p in "${PARTS[@]}"; do
      echo; echo "################ ${CMD}: ${p} ################"
      part "${p}" "${CMD}" 2>&1 | tee "${LOG_DIR}/${CMD}_${p}_${STAMP}.log"
      [[ "${PIPESTATUS[0]}" -eq 0 ]] || { echo "[suite] ${p}: ${CMD} failed"; rc=1; }
    done
    [[ "${CMD}" == "run" ]] && echo && echo "[suite] next: bash $0 metrics && bash $0 summary"
    exit "${rc}" ;;
  vis)
    # re-score from cache (no GPU; refreshes records.json with image paths + full answers), then draw
    VARGS=(--pick "${VIS_PICK:-mix}" --n "${VIS_N:-12}"); [[ -n "${VIS_FILTER:-}" ]] && VARGS+=(--filter "${VIS_FILTER}")
    for p in "${PARTS[@]}"; do
      echo; echo "################ vis: ${p} ################"
      case "${p}" in
        view)     python -u view_chexone.py "${VIEW_ARGS[@]}" --reparse-only > "${LOG_DIR}/vis_rescore_view_${STAMP}.log" 2>&1 ;;
        findings) python -u findings_chexone.py "${FIND_ARGS[@]}" --reparse-only > "${LOG_DIR}/vis_rescore_findings_${STAMP}.log" 2>&1 ;;
        longtail) OUT="${SUITE_DIR}/longtail" PROMPT=mc_reason bash run_cxrlt_eval.sh eval > /dev/null 2>&1 ;;
        phrase)   OUT="${SUITE_DIR}/phrase" PROMPT=reason RUNNER=official bash run_msc_eval.sh eval > /dev/null 2>&1 ;;
        abnormality)
          OUT="${SUITE_DIR}/abnormality" PROMPT=reason RUNNER=official VIS_N="${VIS_N:-2}" bash run_vindr_eval.sh vis
          continue ;;
        *) echo "[suite] unknown part: ${p}"; continue ;;
      esac
      python -u suite_vis.py --part "${p}" --run-dir "${SUITE_DIR}/${p}" "${VARGS[@]}"
    done ;;
  metrics)
    python -u findings_metrics.py --run-dir "${SUITE_DIR}/findings" --bert-model "${BERT_MODEL}" \
        2>&1 | tee "${LOG_DIR}/metrics_${STAMP}.log" ;;
  summary) python -u paper_suite_summary.py --suite-dir "${SUITE_DIR}" 2>&1 | tee "${LOG_DIR}/summary_${STAMP}.log" ;;
  *) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
esac
