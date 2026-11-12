#!/usr/bin/env bash
# =====================================================================
# run_chexone_eval.sh —— CheXOne (StanfordAIMI/CheXOne) 胸片17类检测评测入口
#
# 本目录(track8_chexone/)是自包含的: 脚本、解析、推理全在这里, 不再依赖
# track8_detection_qwen。唯一的外部依赖是同级的 ../track8_detection_sized
# (categories / evaluate / dicom_utils / progress, 由 _shared.py 引入),
# 这样 CheXOne / Qwen3.5 / Faster R-CNN 三条链路仍然共用同一份类别表和
# 同一份官方评分规则, 指标可以直接对拍。
#
# ⚠️ 与 Qwen3.5 的三个结构性差异(详见 chexone_infer.py 文件头):
#   1) 架构走 Qwen2_5_VLForConditionalGeneration, 不是 image-text-to-text
#   2) 输出是 "推理正文 + \boxed{答案}", 不是裸 JSON 数组
#   3) 任务是 per-class 单框 grounding, 且**不吐置信度**
#      -> 用 yes/no VQA 的 P(yes) 当分数兼闸门(--score-mode yesno)
#
# 用法:
#   bash run_chexone_eval.sh check        # 先验权重路径/架构(改路径后第一步)
#   bash run_chexone_eval.sh prompt       # 打印 per-class prompt, 不推理
#   bash run_chexone_eval.sh probe        # 跑 1 张 + 把原始响应 cat 出来看格式
#   bash run_chexone_eval.sh n 50         # 【最常用】随机抽 50 张跑全 17 类(4 卡分片)
#   bash run_chexone_eval.sh smoke        # 20 张试跑, 看解析健康度
#   bash run_chexone_eval.sh full         # 全量 + 官方 S_clin + 逐类 AP/IoU 总表(默认 4 卡分片)
#   bash run_chexone_eval.sh reparse      # 只读缓存重解析重算(不加载模型)
#   bash run_chexone_eval.sh eval         # 只给已有 answer 打分
#   bash run_chexone_eval.sh iou          # 只对已有 answer 出逐类 AP + IoU 表(不占 GPU)
#   bash run_chexone_eval.sh compare      # 三方并排对拍: FRCNN / Qwen3.5 / CheXOne
#   bash run_chexone_eval.sh vis          # 检测框可视化(预测实线 + GT 虚线, 同类同色)
#   bash run_chexone_eval.sh env          # 路径 + 依赖自检
#   bash run_chexone_eval.sh all          # 一条龙: check->prompt->smoke->full->compare->vis
#
# 本目录里的另外两套评测(各有自己的入口脚本):
#   run_vindr_eval.sh   VinDr-CXR 官方 test 上的异常定位(对标论文 mIoU 0.452)
#   run_cxrlt_eval.sh   CXR-LT 长尾病分类(对标论文 accuracy 0.816)
#
# 端到端链路(全在上面这些子命令里, 不用自己拼):
#   DICOM -> PNG(短边 512, 专用目录 png_chexone) -> CheXOne per-class 推理
#         -> Qwen2.5-VL 原生 grounding 标签解析 -> 按类 NMS
#         -> {AccessionNumber}.json -> evaluate.py 官方 S_clin
#         -> 三方对拍表 (+ 可视化)
#
# 覆盖默认值(前缀赋值即可):
#   CHEXONE_MODEL=/真实/路径  bash run_chexone_eval.sh check
#   SCORE_MODE=verbal PRESENT_THR=0.35 bash run_chexone_eval.sh smoke
#   REASONING=1 NSHARD=4 GPUS=0,1,2,3 bash run_chexone_eval.sh full
#   CLASSES=Nodule,Mass  bash run_chexone_eval.sh smoke        # 只跑几类
#   QWEN_ANSWER=/abs/path/answer_qwen FRCNN_ANSWER=/abs/path bash run_chexone_eval.sh compare
#
# ⚠️ 两个口径是 CheXOne 训练时定死的, 已写死在代码里, **没有开关**:
#   1) 坐标归一化到 **0-100**, 不是 Qwen 官方 demo 的 0-1000。原生模板的示例
#      <|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|> 就是 0-100。
#      按 0-1000 解释会缩小 10 倍、面积小 100 倍 —— 症状是"所有框挤在左上角
#      且只有几毫米"。
#   2) 输入图 **先缩到短边 512**(对齐训练时 VinDr PNG 的预处理), 不是
#      max_pixels 预算那套(334 token 的预算会把 512x615 压回 448x560)。
#   想确认口径有没有搞错, 用零成本的几何诊断, 不用重跑模型:
#     python answer_geo.py answer_chexone              # 看面积%/尺寸mm
#     python answer_geo.py answer_chexone --coord-scale 10   # 按 0-100 重解释对比
#   判读: 看「框几何诊断」表 —— 面积%/中心跨度。三个类别以上中心跨度 <0.05
#         = collapse(模型在背先验, 没在看图); 面积 <0.5% = 尺寸过不了 IoU。
# =====================================================================
set -uo pipefail   # 不加 -e: 某步失败也要把后面的提示打完

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}" || exit 1

# ======================= 配置区 (需要改就改这里) =======================
# ---- 数据 / 代码 (本地服务器) ----
IMAGE_ROOT="${IMAGE_ROOT:-/mnt/data/jinghansun/data/match_yibao/reorg}"
SIZED_DIR="${SIZED_DIR:-../track8_detection_sized}"   # 单一事实源(categories/evaluate)

# val 金标准: 按服务器上实际存在的位置自动挑第一个命中的。
# ⚠️ 必须与 Faster R-CNN / Qwen 用的是同一份, 否则指标不可对拍。
VAL_ANN_CANDIDATES=(
  "${SIZED_DIR}/pretrain_splits/val.json"       # run_focal_small.sh / run_evaluate_all.sh 口径
  "${SIZED_DIR}/../pretrain_splits/val.json"    # 顶层 README 步骤2 口径
  "${SIZED_DIR}/splits/val.json"                # 基础版口径
)
# 对拍用的其他两条链路(自动挑; 找不到就只给 CheXOne 自己的分)
FRCNN_CANDIDATES=(
  "${SIZED_DIR}/eval_pretrain_val"              # run_evaluate_all.sh 的 FUSION_OUT
  "${SIZED_DIR}/answer_sized"
)
QWEN_ANSWER="${QWEN_ANSWER:-${HERE}/../track8_detection_qwen/answer_qwen}"

# ---- 模型 ----
# 权重路径(用户确认过的服务器路径; 目录名就是 CheXOne, HF 缓存的默认布局)
export CHEXONE_MODEL="${CHEXONE_MODEL:-/mnt/data/jinghansun/Huggingface/StanfordAIMI/CheXOne}"
GPUS="${GPUS:-0,1,2,3}"
NSHARD="${NSHARD:-4}"                # 3B 单卡装得下, 4 卡各起一进程分片
DTYPE="${DTYPE:-bfloat16}"
FLASH_ATTN="${FLASH_ATTN:-0}"
CONDA_ENV="${CONDA_ENV:-track8}"     # 需含 torch/transformers/qwen_vl_utils/pydicom/pillow

# ---- 输入分辨率 ----
# PNG 由 image_prep 缩到**短边 512**(对齐 CheXOne 训练时 VinDr 的预处理),
# chexone_infer 把 processor 预算放到 640 token, 不会二次缩放。
# 这里不提供开关 —— 口径是训练时定死的。
CHEXONE_PNG="${CHEXONE_PNG:-${HERE}/png_chexone}"   # 尺寸与 Qwen 的 png_qwen 不同, 必须分开
MAX_SIDE="${MAX_SIDE:-1344}"

# ---- 产物 ----
CHEXONE_OUT="${CHEXONE_OUT:-${HERE}/answer_chexone}"
CHEXONE_CACHE="${CHEXONE_CACHE:-${HERE}/cache_chexone}"
LOG_DIR="${LOG_DIR:-./logs/chexone}"

# ---- 推理参数 ----
SCORE_MODE="${SCORE_MODE:-yesno}"    # yesno(推荐) | verbal | const
PRESENT_THR="${PRESENT_THR:-0.5}"    # P(yes) 低于此值就不要框
SCORE_THR="${SCORE_THR:-0.0}"        # 算 mAP 建议 0.0, PR 曲线才完整
REASONING="${REASONING:-0}"          # 1=追加 'reason step by step'(慢, 格式更稳)
ADD_HINT="${ADD_HINT:-0}"            # 1=prompt 带一句类别定义
SMOKE_N="${SMOKE_N:-20}"             # smoke 跑几张
LIMIT="${LIMIT:-0}"                  # >0 则 full/reparse 也只跑 N 张(0=全量)
LIMIT_MODE="${LIMIT_MODE:-random}"   # random=固定种子洗牌后取N(无偏) | head=取前N(偏罕见类)
FORCE="${FORCE:-0}"                  # 1 = 无视已有缓存强制重跑

# ---- 可视化 ----
VIS_OUT="${VIS_OUT:-${HERE}/vis_chexone}"   # vis 输出目录
VIS_N="${VIS_N:-12}"                        # vis 最多画几张
VIS_PNG="${VIS_PNG:-${CHEXONE_PNG}}"        # 底图; 想高清就用 VIS_PNG=<高清 PNG 目录>
# =====================================================================

mkdir -p "${LOG_DIR}" "${CHEXONE_OUT}" "${CHEXONE_CACHE}" "${CHEXONE_PNG}"
STAMP="$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------- conda
locate_conda() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_SH="$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  fi
  if [[ -z "${CONDA_SH:-}" || ! -f "${CONDA_SH}" ]]; then
    for c in "$HOME/miniconda3/etc/profile.d/conda.sh" \
             "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/.conda/etc/profile.d/conda.sh" \
             "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "$c" ]]; then CONDA_SH="$c"; break; fi
    done
  fi
  if [[ -z "${CONDA_SH:-}" || ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: 找不到 conda.sh, 请 export CONDA_SH=/path/to/conda.sh 后重跑"; return 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
}

activate_env() {
  locate_conda || return 1
  conda activate "${CONDA_ENV}" || { echo "ERROR: 无法激活 conda env '${CONDA_ENV}'"; return 1; }
  echo "[env] conda=${CONDA_ENV}  python=$(command -v python)"
}

# ---------------------------------------------------------------- 路径
pick_first() { for p in "$@"; do [[ -e "$p" ]] && { echo "$p"; return 0; }; done; return 1; }

resolve_paths() {
  VAL_ANN="${VAL_ANN:-$(pick_first "${VAL_ANN_CANDIDATES[@]}" || true)}"
  FRCNN_ANSWER="${FRCNN_ANSWER:-$(pick_first "${FRCNN_CANDIDATES[@]}" || true)}"
}

need_val() {   # 需要 val.json 的子命令先过这一关, 免得把空路径传给 python 才炸
  if [[ -z "${VAL_ANN:-}" ]]; then
    echo "ERROR: 找不到 val.json。试过:"
    printf '         %s\n' "${VAL_ANN_CANDIDATES[@]}"
    echo "       请 export VAL_ANN=/abs/path/to/val.json 后重跑"
    return 1
  fi
  [[ -d "${SIZED_DIR}" ]] || { echo "ERROR: 找不到 track8_detection_sized: ${SIZED_DIR}"; return 1; }
  [[ -f "${SIZED_DIR}/evaluate.py" ]] || { echo "ERROR: ${SIZED_DIR}/evaluate.py 缺失(单一事实源)"; return 1; }
  return 0
}

sanity() {
  local bad=0
  echo "----------------------------------------------------------"
  echo "  image_root : ${IMAGE_ROOT}"
  echo "  sized_dir  : ${SIZED_DIR}"
  echo "  val_ann    : ${VAL_ANN:-<未找到>}"
  echo "  frcnn_ans  : ${FRCNN_ANSWER:-<未找到, compare 只比其余两方>}"
  echo "  qwen_ans   : ${QWEN_ANSWER}$( [[ -d "${QWEN_ANSWER}" ]] || echo '   <不存在, compare 会跳过>' )"
  echo "  model      : ${CHEXONE_MODEL}"
  echo "  gpus       : ${GPUS}  nshard=${NSHARD}  dtype=${DTYPE}"
  echo "  out/cache  : ${CHEXONE_OUT}  /  ${CHEXONE_CACHE}  /  ${CHEXONE_PNG}"
  echo "  log_dir    : ${LOG_DIR}"
  echo "----------------------------------------------------------"
  [[ -d "${IMAGE_ROOT}" ]] || { echo "ERROR: image_root 不存在: ${IMAGE_ROOT}"; bad=1; }
  [[ -d "${CHEXONE_MODEL}" ]] || { echo "ERROR: 模型目录不存在: ${CHEXONE_MODEL}"; bad=1; }
  need_val || bad=1
  return ${bad}
}

# ---------------------------------------------------------------- 参数组装
chexone_args() {
  CX_ARGS=(
    --backend local --model "${CHEXONE_MODEL}"
    --dtype "${DTYPE}"
    --ann "${VAL_ANN}" --image-root "${IMAGE_ROOT}"
    --out "${CHEXONE_OUT}" --cache-dir "${CHEXONE_CACHE}" --png-dir "${CHEXONE_PNG}"
    --score-mode "${SCORE_MODE}" --present-thr "${PRESENT_THR}"
    --max-side "${MAX_SIDE}" --score-thr "${SCORE_THR}"
    --limit-mode "${LIMIT_MODE}" --gt-ann "${VAL_ANN}"
  )
  [[ "${REASONING}"  == "1" ]] && CX_ARGS+=(--reasoning)
  [[ "${FLASH_ATTN}" == "1" ]] && CX_ARGS+=(--flash-attn)
  [[ "${ADD_HINT}"   == "1" ]] && CX_ARGS+=(--add-hint)
  [[ "${FORCE}"      == "1" ]] && CX_ARGS+=(--overwrite-cache)
  [[ -n "${CLASSES:-}" ]]     && CX_ARGS+=(--classes "${CLASSES}")
  return 0
}

chexone_iou_report() {
  # 逐类 AP(官方) + GT 视角 IoU; 结果同时落 csv/json 方便贴表
  local prefix="${LOG_DIR}/chexone_iou_${STAMP}"
  python -u iou_report.py --pred-dir "${CHEXONE_OUT}" --gt-ann "${VAL_ANN}" \
      --image-root "${IMAGE_ROOT}" --out-prefix "${prefix}" \
      2>&1 | tee "${prefix}.log"
}

# ---------------------------------------------------------------- dispatch
resolve_paths
cmd="${1:-help}"

case "${cmd}" in
  check)
    # 路径手打错是最常见的事故(多一个少一个字母就白跑一小时), 所以单独一个子命令先验
    echo "[check] CHEXONE_MODEL = ${CHEXONE_MODEL}"
    if [[ ! -d "${CHEXONE_MODEL}" ]]; then
      echo "[check] ✗ 目录不存在。确认一下:"
      echo "        ls /mnt/data/jinghansun/Huggingface/StanfordAIMI/"
      echo "        也可以 CHEXONE_MODEL=<真实路径> bash $0 check"
      exit 1
    fi
    echo "[check] ✓ 目录存在"
    for f in config.json generation_config.json preprocessor_config.json; do
      if [[ -f "${CHEXONE_MODEL}/${f}" ]]; then
        echo "[check]   ✓ ${f}"
      else
        echo "[check]   · ${f} 缺失(不致命, 但 tokenizer/生成配置可能要靠默认)"
      fi
    done
    if [[ -f "${CHEXONE_MODEL}/config.json" ]]; then
      echo "[check] config 摘要:"
      grep -E '"(model_type|architectures|torch_dtype)"' "${CHEXONE_MODEL}/config.json" || true
    fi
    n_safetensors="$(find "${CHEXONE_MODEL}" -name '*.safetensors' 2>/dev/null | wc -l)"
    n_bin="$(find "${CHEXONE_MODEL}" -name '*.bin' 2>/dev/null | wc -l)"
    echo "[check] 权重文件: safetensors=${n_safetensors} bin=${n_bin}"
    if [[ "${n_safetensors}" -eq 0 && "${n_bin}" -eq 0 ]]; then
      echo "[check] ✗ 没找到权重文件, 下载完整了吗?"
      echo "        huggingface-cli download StanfordAIMI/CheXOne \\"
      echo "            --local-dir ${CHEXONE_MODEL}"
      exit 1
    fi
    echo "[check] 下一步: bash $0 probe"
    ;;
  env)
    sanity && echo "[env] 路径自检通过"
    activate_env && python -u - <<'PY'
import sys
ok = True
for mod, why in (("numpy", "必需"), ("pydicom", "读DICOM"), ("PIL", "导出PNG"),
                 ("torch", "推理"), ("transformers", "推理(需含 Qwen2_5_VL)"),
                 ("qwen_vl_utils", "官方 runner 必需"),
                 ("matplotlib", "vis / vindr_vis 画图"),
                 ("tqdm", "进度条, 可缺")):
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", "?")
        print(f"[env] OK   {mod:<16s} {v}")
    except Exception as e:
        hard = mod in ("numpy", "pydicom", "PIL", "torch", "transformers", "qwen_vl_utils")
        print(f"[env] {'MISS' if hard else 'warn'} {mod:<16s} ({why}) -> {e}")
        if hard:
            ok = False
try:
    import torch
    print(f"[env] cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}")
except Exception:
    pass
sys.exit(0 if ok else 1)
PY
    ;;
  prompt)
    activate_env || exit 1
    chexone_args
    python -u chexone_infer.py --dry-run
    ;;
  smoke)
    need_val || exit 1
    activate_env || exit 1
    chexone_args
    # ⚠️ 这里是 case 分支, **不是函数** —— 用 local 会直接报
    #    "local: can only be used in a function", 再把后面的 $log 变成 unbound。
    #    case 分支里一律用普通赋值(变量名加 cx 前缀避免和外层的 log 打架)。
    cx_log="${LOG_DIR}/chexone_smoke_${STAMP}.log"
    echo "[smoke] ${SMOKE_N} 张 -> ${cx_log}"
    CUDA_VISIBLE_DEVICES="${GPUS}" python -u chexone_infer.py "${CX_ARGS[@]}" \
        --limit "${SMOKE_N}" 2>&1 | tee "${cx_log}"
    echo
    echo "[smoke] 先看『解析健康度』:"
    echo "   \\boxed 命中率低     -> 加 REASONING=1(论文说 reasoning 模式才稳守格式)"
    echo "   零检出类别多        -> PRESENT_THR=0.35 放宽闸门再看"
    echo "   (口径已按 CheXOne 训练约定固定: 坐标 0-100 / 输入短边 512, 不要再调)"
    ;;
  n)
    # 便捷入口: bash run_chexone_eval.sh n 50
    # == 随机抽 50 张跑全 17 类, 多卡各起一进程分片, 跑完自动合并重算。
    cx_n="${2:-50}"
    if ! [[ "${cx_n}" =~ ^[0-9]+$ ]] || [[ "${cx_n}" -le 0 ]]; then
      echo "[n] 用法: bash $0 n <张数>   例: bash $0 n 50"; exit 1
    fi
    export LIMIT="${cx_n}"
    echo "[n] 随机抽 ${cx_n} 张, 全 17 类, GPUS=${GPUS} NSHARD=${NSHARD}"
    echo "[n] 盯进度: tail -f ${LOG_DIR}/chexone_full_*_s0.log"
    exec bash "$0" full
    ;;
  probe)
    # 跑 1 张, 然后把原始响应原样 cat 出来 —— 用来确认它到底吐的是
    # \boxed{[x1,y1,x2,y2]}、裸 JSON、还是一整段话。这一步比看 mAP 重要得多:
    # 格式没搞清楚就上全量, 跑完几小时才发现 bad_box 90% 就太贵了。
    SMOKE_N=1 NSHARD=1 bash "$0" smoke || exit 1
    N="${PROBE_N:-8}"
    echo
    echo "[probe] 最近 ${N} 条 localize 原始响应 (${CHEXONE_CACHE}):"
    hit=0
    for f in $(ls -t "${CHEXONE_CACHE}"/*__box_*.txt 2>/dev/null | head -n "${N}"); do
      echo "---------------- $(basename "${f}")"
      cat "${f}"; echo
      hit=$((hit + 1))
    done
    if [[ "${hit}" -eq 0 ]]; then
      echo "  (没有 box_* 缓存 —— 要么全被 PRESENT_THR 挡了, 要么上面那步跑挂了)"
    fi
    echo "[probe] 最近 ${N} 条 yes/no 原始响应:"
    for f in $(ls -t "${CHEXONE_CACHE}"/*__yn_*.txt 2>/dev/null | head -n "${N}"); do
      echo "---------------- $(basename "${f}")"
      cat "${f}"; echo
    done
    echo "[probe] 判读提示: 出现 <|ref|>..<|box|>(x1,y1),(x2,y2) 或 \\boxed{...} 即格式正常;"
    echo "                  全是整段话 -> 加 REASONING=1 再 probe 一次"
    ;;
  full)
    need_val || exit 1
    activate_env || exit 1
    chexone_args
    [[ "${LIMIT}" -gt 0 ]] && CX_ARGS+=(--limit "${LIMIT}")
    if [[ "${NSHARD}" -le 1 ]]; then
      cx_log="${LOG_DIR}/chexone_full_${STAMP}.log"
      CUDA_VISIBLE_DEVICES="${GPUS}" python -u chexone_infer.py "${CX_ARGS[@]}" \
          2>&1 | tee "${cx_log}"
    else
      # 3B 单卡就装得下, 4×H20 各起一个进程做数据分片最划算。
      # 缓存按 acc+类 命名, 多进程写同一个目录不会互相踩。
      IFS=',' read -ra _GA <<< "${GPUS}"
      if [[ "${#_GA[@]}" -lt "${NSHARD}" ]]; then
        echo "[full] ERROR: NSHARD=${NSHARD} 但 GPUS 只有 ${#_GA[@]} 个: ${GPUS}"; exit 1
      fi
      cx_pids=(); cx_i=0; cx_log="${LOG_DIR}/chexone_full_${STAMP}"
      for ((cx_i = 0; cx_i < NSHARD; cx_i++)); do
        CUDA_VISIBLE_DEVICES="${_GA[cx_i]}" python -u chexone_infer.py "${CX_ARGS[@]}" \
            --shard "${cx_i}" --n-shard "${NSHARD}" --no-eval \
            > "${cx_log}_s${cx_i}.log" 2>&1 &
        cx_pids+=($!)
      done
      echo "[full] ${NSHARD} 个分片进程已起: ${cx_log}_s*.log"
      echo "[full] 盯进度:  tail -f ${cx_log}_s0.log"
      wait "${cx_pids[@]}"
      echo "[full] 分片跑完 -> 合并重算全量指标"
      python -u chexone_infer.py "${CX_ARGS[@]}" --reparse-only \
          2>&1 | tee "${cx_log}_reparse.log"
    fi
    # 跑完直接出 AP + IoU 总表(官方同口径), 不用再敲一条命令
    chexone_iou_report
    ;;
  reparse)
    need_val || exit 1
    activate_env || exit 1
    chexone_args
    [[ "${LIMIT}" -gt 0 ]] && CX_ARGS+=(--limit "${LIMIT}")
    python -u chexone_infer.py "${CX_ARGS[@]}" --reparse-only \
        2>&1 | tee "${LOG_DIR}/chexone_reparse_${STAMP}.log"
    ;;
  iou)
    need_val || exit 1
    activate_env || exit 1
    chexone_iou_report
    ;;
  eval)
    need_val || exit 1
    activate_env || exit 1
    chexone_args
    python -u chexone_infer.py "${CX_ARGS[@]}" --reparse-only --no-eval > /dev/null 2>&1
    ( cd "${SIZED_DIR}" && python -u evaluate.py \
        --pred-dir "$(cd "${CHEXONE_OUT}" && pwd)" \
        --gt-ann "$(cd "$(dirname "${VAL_ANN}")" && pwd)/$(basename "${VAL_ANN}")" \
        --image-root "${IMAGE_ROOT}" ) 2>&1 | tee "${LOG_DIR}/chexone_eval_${STAMP}.log"
    ;;
  compare)
    # 三方并排对拍(总表 + 逐类 AP + Δ)。FRCNN / Qwen 谁的结果在就比谁,
    # 只有 CheXOne 自己时也给分, 不至于空转。
    need_val || exit 1
    activate_env || exit 1
    cx_runs=()
    [[ -n "${FRCNN_ANSWER:-}" && -d "${FRCNN_ANSWER}" ]] && cx_runs+=("frcnn=${FRCNN_ANSWER}")
    [[ -d "${QWEN_ANSWER}" ]]                            && cx_runs+=("qwen=${QWEN_ANSWER}")
    cx_runs+=("chexone=${CHEXONE_OUT}")
    if [[ "${#cx_runs[@]}" -lt 2 ]]; then
      echo "[compare] ⚠️ 只有 CheXOne 一份结果。"
      echo "        export FRCNN_ANSWER=/abs/path/to/answer_dir  和/或  QWEN_ANSWER=/abs/path 后重跑才能出 Δ"
    fi
    echo "[compare] runs: ${cx_runs[*]}"
    python -u compare_answers.py \
        --gt-ann "${VAL_ANN}" --image-root "${IMAGE_ROOT}" --common-only \
        --json-out "${LOG_DIR}/chexone_compare_${STAMP}.json" \
        --runs "${cx_runs[@]}" 2>&1 | tee "${LOG_DIR}/chexone_compare_${STAMP}.log"
    ;;
  vis)
    need_val || exit 1
    activate_env || exit 1
    # 底图默认用 png_chexone(短边 512) —— 那是模型真正看到的图, 能看出小病灶
    # 在输入分辨率下到底还看不看得见。想看清楚就换高清底图:
    #   VIS_PNG=/path/to/png_qwen bash run_chexone_eval.sh vis
    # 坐标是按 answer 里 ImageSize 等比缩放的, 换底图不会画错位。
    mkdir -p "${VIS_OUT}" "${VIS_PNG}"
    python -u visualize_detections.py \
        --answer-dir "${CHEXONE_OUT}" --png-dir "${VIS_PNG}" --out-dir "${VIS_OUT}" \
        --gt-ann "${VAL_ANN}" --with-gt --n "${VIS_N}" --tag "CheXOne" \
        2>&1 | tee "${LOG_DIR}/chexone_vis_${STAMP}.log"
    echo "[vis] -> ${VIS_OUT}  (预测=实线, GT=虚线, 同类同色; 底图=${VIS_PNG})"
    ;;
  all)
    # 真·一条龙: 验权重 -> prompt -> smoke -> 全量(带 S_clin) -> 三方对拍 -> 可视化
    bash "$0" check   || exit 1
    bash "$0" prompt
    bash "$0" smoke   || exit 1
    echo
    echo "=========================================================="
    echo " [all] smoke 结束。看一眼『解析健康度』再继续:"
    echo "   \\boxed 命中率低 -> REASONING=1 重跑(走原生 reasoning 模板)"
    echo "   零检出类别多    -> PRESENT_THR=0.35 放宽闸门"
    echo "   框全挤在角落    -> python answer_geo.py ${CHEXONE_OUT} 看面积%/尺寸mm"
    read -r -p " [all] 继续跑全量? [y/N] " _go
    [[ "${_go:-N}" =~ ^[Yy]$ ]] || { echo " 已停。确认后单独跑: bash $0 full"; exit 0; }
    bash "$0" full    || exit 1
    bash "$0" compare || true
    bash "$0" vis     || true
    echo
    echo "=========================================================="
    echo " ALL DONE"
    echo "   answer  : ${CHEXONE_OUT}/      (每图一个 {AccessionNumber}.json)"
    echo "   cache   : ${CHEXONE_CACHE}/    (原始响应, 改参数后 reparse 不重跑模型)"
    echo "   对拍表  : ${LOG_DIR}/chexone_compare_*.log"
    echo "   可视化  : ${VIS_OUT}/"
    echo "=========================================================="
    ;;
  help|*)
    sed -n "2,$(( $(grep -n '^set -uo pipefail' "$0" | head -1 | cut -d: -f1) - 1 ))p" "$0" \
      | sed 's/^# \{0,1\}//'
    echo
    echo "当前配置:"
    echo "  CHEXONE_MODEL          = ${CHEXONE_MODEL}"
    echo "  CHEXONE_OUT / CACHE    = ${CHEXONE_OUT} / ${CHEXONE_CACHE}"
    echo "  SCORE_MODE / PRESENT_THR = ${SCORE_MODE} / ${PRESENT_THR}"
    echo "  CHEXONE_PNG            = ${CHEXONE_PNG} (短边 512, 对齐训练预处理)"
    echo "  坐标口径               = 0-100 (写死, 无开关)"
    echo "  REASONING / ADD_HINT   = ${REASONING} / ${ADD_HINT}"
    echo "  GPUS / NSHARD          = ${GPUS} / ${NSHARD}"
    echo "  CONDA_ENV              = ${CONDA_ENV}"
    echo "  VAL_ANN                = ${VAL_ANN:-<未找到>}"
    exit 1 ;;
esac
