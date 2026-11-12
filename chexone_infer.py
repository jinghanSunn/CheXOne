"""
CheXOne 胸片 17 类检测推理 —— 与 qwen_infer.py **同口径**出 {AccessionNumber}.json,
可直接喂给 track8_detection_sized/evaluate.py 拿官方 S_clin, 跟 Faster R-CNN /
Qwen3.5 三条链路对拍。

为什么不能只是"把 --model 换成 CheXOne":
  1. **架构**: CheXOne 是 Qwen2.5-VL-3B 的后训练, 走 Qwen2_5_VLForConditionalGeneration,
     不是 Qwen3.5 的 image-text-to-text。local_runner 已加多入口回退。
  2. **输出格式**: 原生 grounding 标签 <|ref|>name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>,
     坐标 **0-100**; reasoning 模式再包一层 \\boxed{}。parse_utils.extract_grounding 负责。
  3. **任务形态**: 「给定异常名 -> 定位」, **不带置信度**。所以主循环是 per-class:
     每个类别先问一个 yes/no 拿 P(yes) 当分数兼闸门, 超过阈值才去要框。
     类别由**问题**决定, 不采信 <|ref|> 里写的病名(否则跨类串标)。
  4. **输入**: PNG 先缩到**短边 512**(对齐训练时 VinDr 的预处理), processor 预算
     放到 640 token 保证不再二次缩放。

置信度为什么必须单独拿:
  mAP 是按置信度排序算 PR 曲线的。CheXOne 只吐坐标不吐分数, 全给同一个分 = 随机
  排序 = AP 严重失真; 而且 per-class 每类最多一个框, 没有框间排序信息可借用。
  所以走 yes/no VQA 的 P(yes)(--score-mode yesno, 默认)。这也是我们之前排的
  改进清单里第 2 条(logprob/yes-no 拿排序分), 这里顺手验证。

用法(本地服务器, 数据在 /mnt/data/jinghansun/data/match_yibao/reorg):
  # 1) 先 20 张试跑, 重点看坐标尺度分布和 \boxed 命中率
  CUDA_VISIBLE_DEVICES=0 python -u chexone_infer.py --limit 20
  # 2) 全量, 4 张卡各起一个进程做分片(缓存按 acc+类 命名, 不会互相踩)
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -u chexone_infer.py \
      --shard $i --n-shard 4 > log_shard$i.txt 2>&1 & done
  # 3) 只重解析(不占 GPU)
  python -u chexone_infer.py --reparse-only
"""
import os
import sys
import json
import time
import random as _random
import hashlib
import argparse

import _shared  # noqa: F401  (bootstrap sys.path)
from _shared import _cat
from progress import pbar, is_tqdm
from evaluate import nms_per_class
from effusion_infer import effusion_grade_probs

from image_prep import dicom_to_png_file, list_images_from_ann
from parse_utils import parse_detections, extract_boxed
from local_runner import QwenLocalRunner, GenerationConfig
from qwen_client import add_client_args, resolve_model
from prompt_builder import CLASS_HINT

NUM_CLASSES = _cat.NUM_CLASSES
CATID_TO_EN = _cat.CATID_TO_EN
CATID_TO_CN = _cat.CATID_TO_CN
EN_TO_CATID = _cat.EN_TO_CATID
MASS_EN = _cat.MASS_EN
PLEURAL_EFFUSION_EN = _cat.PLEURAL_EFFUSION_EN
FUSION_NMS_IOU = _cat.FUSION_NMS_IOU
build_finding_probabilities = _cat.build_finding_probabilities
make_grade_probabilities = _cat.make_grade_probabilities

# --------------------------------------------------------------------------
# 类别 -> 自然语言短语。CheXOne 训练时见的异常名是 VinDr-CXR 那种小写短语
# ("aortic enlargement" / "pleural effusion"), 直接喂 CamelCase 的 key 是
# off-distribution。这里逐个手工对齐, 不要偷懒用正则切词。
PHRASE_OVERRIDE = {
    "Nodule": "pulmonary nodule",
    "Mass": "pulmonary mass",
    "Cavity": "lung cavity",
    "Edema": "pulmonary edema",
    "MediastinalWidening": "mediastinal widening",
    "AirBronchogram": "air bronchogram",
    "Fibrosis": "pulmonary fibrosis",
    "Calcification": "pulmonary calcification",
    "Infiltration": "pulmonary infiltration",
}


def _phrase(en):
    if en in PHRASE_OVERRIDE:
        return PHRASE_OVERRIDE[en]
    out = []
    for i, ch in enumerate(en):
        if ch.isupper() and i > 0:
            out.append(" ")
        out.append(ch.lower())
    return "".join(out)


PHRASE = {en: _phrase(en) for en in EN_TO_CATID}

_BOXED_INSTR = "Put your final answer within \\boxed{}."

# --------------------------------------------------------------------------
# CheXOne 的 grounding 用的是 **Qwen2.5-VL 原生模板**, 不是我们自己编的问法。
# 两个模式(与官方 README 的 Reasoning / Instruct 两档一一对应):
#
#   Instruct  : "Detect {ph} in the given image."
#               没有格式说明 —— 模型直接吐原生标签
#               <|ref|>name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>
#
#   Reasoning : 同上 + Output format / Example / "reason step by step" +
#               "put your final answer within \boxed{}"
#
# ⚠️ 两个关键口径(都是训练时定死的, 改了就 off-distribution):
#   1) 坐标归一化到 **0-100**, 不是 Qwen 官方 demo 里的 0-1000。示例里的
#      (12,60),(18,87) 就是 0-100。按 0-1000 解释会缩小 10 倍、面积小 100 倍,
#      症状就是"所有框挤在左上角且只有几毫米"。
#   2) 训练图是 VinDr 的 PNG **先缩到短边 512** 再标注的 —— 所以送入模型的图
#      也要短边 512(见 --short-side), 不是 max_pixels 预算那套。
# --------------------------------------------------------------------------
_GROUNDING_EXAMPLE = (
    "<|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>")

_LOCALIZE_INSTRUCT = "Detect {ph} in the given image.{hint}"

_LOCALIZE_REASON = """\
Detect {ph} in the given image.
{hint}Output format:
<|ref|>target name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>
Example: {example}
Please reason step by step, and put your final answer within \\boxed{{}}."""

_YESNO_TMPL = "Is {ph} present in this chest X-ray? Answer only yes or no.{hint}"

# verbal 模式只用于对照实验(让模型一轮自报 present+confidence+bbox)。
# CheXOne 没被这么训过, 格式依从性存疑, 默认不走这条路。
_VERBAL_TMPL = """\
Look at this chest X-ray and assess: {ph}.
{hint}Return a single JSON object, exactly:
  {{"present": true|false, "confidence": <float 0~1>, "bbox_2d": [x1, y1, x2, y2]}}
Coordinates are normalized to 0-100 over the whole image (Qwen2.5-VL grounding \
convention). Set "bbox_2d" to [] when "present" is false. {boxed}"""


def build_prompts(en, mode, reasoning, add_hint):
    ph = PHRASE[en]
    hint = f"(Reference: {CLASS_HINT[en]})\n" if (add_hint and en in CLASS_HINT) else ""
    if mode == "verbal":
        boxed = _BOXED_INSTR + (" Please reason step by step." if reasoning else "")
        return _VERBAL_TMPL.format(ph=ph, hint=hint, boxed=boxed)
    if reasoning:
        loc = _LOCALIZE_REASON.format(ph=ph, hint=hint, example=_GROUNDING_EXAMPLE)
    else:
        loc = _LOCALIZE_INSTRUCT.format(ph=ph, hint=hint)
    return (_YESNO_TMPL.format(ph=ph, hint=hint), loc)


# --------------------------------------------------------------------------
def laterality_from_bbox(cx_ratio):
    """与 inference_sized.py / qwen_infer.py 完全一致的侧别启发式。"""
    if cx_ratio < 0.4:
        return {"left": 0.8, "right": 0.1, "bilateral": 0.1}
    if cx_ratio > 0.6:
        return {"left": 0.1, "right": 0.8, "bilateral": 0.1}
    return {"left": 0.2, "right": 0.2, "bilateral": 0.6}


def _cpath(cache_dir, acc, tag):
    return os.path.join(cache_dir, f"{acc}__{tag}.txt")


# yes/no 打分实现的版本号。改了打分方式(比如从 generate scores 改成原始 logits)
# 就 +1, 旧缓存自动失效。
_YN_IMPL = "rawlogit1"


def _fp(*parts):
    return hashlib.md5("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8]


def _input_fp(args):
    """影响「模型看到什么」的一切: 预处理 + vision token 预算。"""
    return (args.short_side, args.max_side, args.clahe,
            args.min_pixels_tokens, args.max_pixels_tokens)


def _box_tag(en, args):
    """定位缓存 key = 类别 + (实际 prompt 原文 + 输入口径) 的指纹。

    之前用固定的 box_n_{en}: instruct / REASONING=1 / ADD_HINT=1 三种 prompt 共用
    同一个文件, 切换后**静默命中**上一次的输出 —— 看着跑了其实没跑。直接对 prompt
    原文取指纹, 以后改模板也不用记得手动换 key。
    """
    if args.score_mode == "verbal":
        pr = build_prompts(en, "verbal", args.reasoning, args.add_hint)
    else:
        pr = build_prompts(en, "yesno", args.reasoning, args.add_hint)[1]
    return f"box_{en}_{_fp(pr, *_input_fp(args))}"


def _yn_tag(en, args):
    """yes/no 缓存 key。之前是固定的 yn_{en}, 从 334 token / png_qwen 那批一直沿用
    到现在 —— 换成短边 512 后 P(yes) 仍读旧图上的值。"""
    q = build_prompts(en, "yesno", args.reasoning, args.add_hint)[0]
    return f"yn_{en}_{_fp(q, _YN_IMPL, *_input_fp(args))}"


def _geo_add(geo, en, box, W0, H0, ps_col, ps_row):
    """累计框几何 —— 判 model collapse 用。

    只看 mAP 会得出「模型不行」这种没法行动的结论。看面积%+中心跨度能区分:
      - 面积正常、中心分散  -> 定位不准, 还有救(分辨率/prompt/后处理)
      - 面积 <0.5% 且中心挤在一点 -> **collapse**, 模型根本没在看图, 换 prompt 试试
      - 中心分散但值域不对 -> 坐标尺度问题, 改 --coord-mode
    """
    if geo is None:
        return
    x1, y1, x2, y2 = box
    g = geo.setdefault(en, {"n": 0, "a": 0.0, "cx": 0.0, "cy": 0.0,
                            "wmm": 0.0, "hmm": 0.0,
                            "x0": 1e9, "x1": -1e9, "y0": 1e9, "y1": -1e9})
    cx = (x1 + x2) / 2.0 / max(1.0, W0)
    cy = (y1 + y2) / 2.0 / max(1.0, H0)
    g["n"] += 1
    g["a"] += (x2 - x1) * (y2 - y1) / max(1.0, W0 * H0)
    g["cx"] += cx; g["cy"] += cy
    g["wmm"] += (x2 - x1) * ps_col
    g["hmm"] += (y2 - y1) * ps_row
    g["x0"] = min(g["x0"], cx); g["x1"] = max(g["x1"], cx)
    g["y0"] = min(g["y0"], cy); g["y1"] = max(g["y1"], cy)


def build_result(acc, packed, info, args):
    """packed(已 NMS 的 [{'box','en','conf'}]) -> 与 qwen_infer 逐字段一致的 spec。"""
    W0, H0 = info["W0"], info["H0"]
    ps_col, ps_row = info["ps_col"], info["ps_row"]
    detections, key_findings = [], []
    for rank, p in enumerate(packed):
        x1, y1, x2, y2 = p["box"]
        en, cid, sc = p["en"], p["cid"], p["conf"]
        x1mm, x2mm = x1 * ps_col, x2 * ps_col
        y1mm, y2mm = y1 * ps_row, y2 * ps_row
        prob_vec = [0.01] * NUM_CLASSES
        prob_vec[cid] = float(sc)
        cx_ratio = ((x1 + x2) / 2.0) / max(1.0, W0)
        det = {
            "LesionID": f"DET_{rank + 1:03d}",
            "BBox_mm": [round(float(x1mm), 2), round(float(y1mm), 2),
                        round(float(x2mm), 2), round(float(y2mm), 2)],
            "Finding": en,
            "Confidence": round(float(sc), 4),
            "Finding_probabilities": {
                k: round(v, 4) for k, v in build_finding_probabilities(prob_vec).items()},
            "Laterality_probabilities": laterality_from_bbox(cx_ratio),
        }
        if en == MASS_EN:
            # 与 FRCNN 基线同口径: 用框长边几何估计, 不采信模型自报
            det["AdditionalAttributes"] = {
                "MaxDiameter_mm": round(float(max(x2mm - x1mm, y2mm - y1mm)), 2)}
        elif en == PLEURAL_EFFUSION_EN:
            if args.effusion_null:
                det["AdditionalAttributes"] = {"EffusionGrade": None}
            else:
                predicted, probabilities = effusion_grade_probs([x1, y1, x2, y2], H0)
                det["AdditionalAttributes"] = {"EffusionGrade": {
                    "predicted": predicted, "probabilities": probabilities}}
        detections.append(det)
        key_findings.append(en)
    conclusion = (f"共检出{len(detections)}个病灶：" + "、".join(key_findings) + "。") \
        if detections else "双肺未见明确异常征象。"
    return {
        "AccessionNumber": acc,
        "IsNotHumanBodyProb": 0.0,
        "IsStitchedProb": 0.0,
        "ProcessingTime_ms": 0,
        "Prediction": {
            "ImageSize": {"width_px": W0, "height_px": H0,
                          "pixel_spacing_mm": [round(ps_col, 4), round(ps_row, 4)]},
            "Detections": detections,
            "Interpretation": {"Conclusion": conclusion, "KeyFindings": key_findings},
        },
    }


# --------------------------------------------------------------------------
def infer_one(runner, dcm_path, class_ens, args, stats, geo=None):
    png_path, info = dicom_to_png_file(
        dcm_path, args.png_dir, max_side=args.max_side, clahe=args.clahe,
        overwrite=args.overwrite_png, short_side=args.short_side)
    W0, H0, W_in, H_in = info["W0"], info["H0"], info["W_in"], info["H_in"]
    acc = info["acc"]
    raw_dets, present = [], {}

    for en in class_ens:
        cid = EN_TO_CATID[en]
        if args.score_mode == "yesno":
            cp = _cpath(args.cache_dir, acc, _yn_tag(en, args)) if args.cache_dir else None
            p, txt, mass = None, None, None
            if cp and os.path.exists(cp) and not args.overwrite_cache:
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    p, txt, mass = rec.get("p"), rec.get("text"), rec.get("mass")
                except Exception:
                    p = None
            if p is None:
                if args.reparse_only:
                    stats["cache_miss"] = stats.get("cache_miss", 0) + 1
                    continue
                q, _ = build_prompts(en, "yesno", args.reasoning, args.add_hint)
                p, txt = runner.yesno_score(png_path, q)
                mass = runner.last_yesno_mass
                if cp:
                    with open(cp, "w", encoding="utf-8") as f:
                        json.dump({"p": p, "text": txt, "mass": mass}, f,
                                  ensure_ascii=False)
            if mass is not None and mass < 0.5:
                stats["yn_low_mass"] = stats.get("yn_low_mass", 0) + 1
            stats["yn_query"] = stats.get("yn_query", 0) + 1
            present[en] = p
            if p < args.present_thr:
                stats["filtered_by_present"] = stats.get("filtered_by_present", 0) + 1
                continue

        tag = _box_tag(en, args)
        cp = _cpath(args.cache_dir, acc, tag) if args.cache_dir else None
        txt = None
        if cp and os.path.exists(cp) and not args.overwrite_cache:
            with open(cp, "r", encoding="utf-8") as f:
                txt = f.read()
        elif args.reparse_only:
            stats["cache_miss"] = stats.get("cache_miss", 0) + 1
            continue
        stats["box_query"] = stats.get("box_query", 0) + 1
        if txt is None:
            if args.score_mode == "verbal":
                pr = build_prompts(en, "verbal", args.reasoning, args.add_hint)
            else:
                pr = build_prompts(en, "yesno", args.reasoning, args.add_hint)[1]
            txt = runner.chat(png_path, pr)
            if cp:
                with open(cp, "w", encoding="utf-8") as f:
                    f.write(txt or "")

        local = {}
        dets = parse_detections(txt, W_in, H_in, W0, H0, coord_mode=args.coord_mode,
                                stats=local, default_label=en)
        if extract_boxed(txt):
            stats["boxed_hit"] = stats.get("boxed_hit", 0) + 1
        else:
            stats["no_boxed"] = stats.get("no_boxed", 0) + 1
        for k, v in local.items():
            stats[k] = stats.get(k, 0) + v
        # 置信度: 有 yes/no 分数就用它, 否则用模型自报, 再否则交给 parse 的伪分数
        for d in dets:
            if args.score_mode == "yesno":
                d["conf"] = float(present.get(en, 1.0))
            elif args.score_mode == "const":
                d["conf"] = 1.0
            _geo_add(geo, en, d["box_px"], W0, H0, info["ps_col"], info["ps_row"])
        raw_dets.extend(dets)

    kept = [d for d in raw_dets if d["conf"] >= args.score_thr]
    packed = [{"box": d["box_px"], "en": d["en"], "conf": d["conf"], "cid": d["cid"]}
              for d in kept]
    packed = nms_per_class(packed, args.nms_iou)
    packed.sort(key=lambda p: -p["conf"])
    if args.max_det > 0:
        packed = packed[:args.max_det]
    return build_result(acc, packed, info, args), acc, present


# --------------------------------------------------------------------------
def _print_health(stats, n_img, n_cls):
    print("\n" + "=" * 62)
    print("解析健康度 (这张表比 mAP 更先看 —— 解析崩了 mAP 没意义)")
    print("=" * 62)
    total = max(1, n_img * n_cls)
    nq = max(1, stats.get("box_query", 0))
    nyn = max(1, stats.get("yn_query", 0))
    reasoning = bool(stats.get("reasoning", 0))
    rows = [("类别查询次数", total, None),
            ("实发定位请求", stats.get("box_query", 0), total),
            ("原生 <|box|> 命中", stats.get("grounding_hit", 0), nq),
            ("  └ 只剩裸 (x,y),(x,y)", stats.get("gsrc_text_tuple", 0)
             + stats.get("gsrc_boxed_tuple", 0), nq),
            ("  └ 推理正文取末框", stats.get("gsrc_text_last", 0), nq),
            ("<|ref|> 与查询类不符", stats.get("ref_mismatch", 0), nq),
            ("复述 prompt 示例框", stats.get("example_echo", 0), nq)]
    if reasoning:
        rows += [("\\boxed 命中", stats.get("boxed_hit", 0), nq),
                 ("\\boxed 缺失", stats.get("no_boxed", 0), nq)]
    rows += [("JSON/坐标抽取失败", stats.get("no_json", 0), nq),
             ("坐标非法/退化框", stats.get("bad_box", 0), nq),
             ("yes/no 质量<0.5", stats.get("yn_low_mass", 0), nyn),
             ("被 present 闸门挡掉", stats.get("filtered_by_present", 0), total),
             ("缓存缺失(reparse-only)", stats.get("cache_miss", 0), total)]
    for k, v, den in rows:
        pct = f"{100.0 * v / den:5.1f}%" if den else ""
        print(f"  {k:<24s}{v:>8d}  {pct}")
    n_ground = stats.get("grounding_hit", 0)
    if n_ground < 0.9 * nq:
        print(f"  ⚠️ 原生 grounding 命中率只有 {100.0 * n_ground / nq:.0f}% —— "
              "cat 一个 cache 里的 box_*.txt 看模型到底吐了什么。")
    if stats.get("ref_mismatch", 0) > 0.2 * nq:
        print("  ⚠️ >20% 的回答在 <|ref|> 里写了别的病名 —— 模型在答「这里有什么」"
              "而不是「X 在哪」。框已按查询类记账, 但这说明类间判别弱。")
    if stats.get("yn_low_mass", 0) > 0.1 * nyn:
        print("  ⚠️ >10% 的 yes/no 首 token 不是 yes/no —— 这些 P(yes) 是噪声, "
              "闸门不可信。")
    if reasoning and stats.get("no_boxed", 0) > 0.1 * nq:
        print("  ⚠️ \\boxed 缺失率 >10% —— reasoning 模式要求最终答案进 \\boxed, "
              "缺失说明格式依从性不够或 max_new_tokens 不够。")
    modes = {k[5:]: v for k, v in stats.items() if k.startswith("mode_")}
    if modes:
        print("\n  坐标尺度判定分布:", "  ".join(f"{m}={c}" for m, c in sorted(modes.items())))
        if len(modes) > 1:
            print("  ⚠️ 出现多种尺度 —— 坐标口径写死了 norm100, 不该出现别的;"
                  " 检查是不是有裸 JSON/绝对像素的旧格式混进来了")
    print("=" * 62)


def _print_geometry(geo):
    """框几何诊断 —— 判断是「定位不准」还是「模型 collapse」。"""
    if not geo:
        return
    print("\n" + "=" * 78)
    print("框几何诊断 (collapse 检测: 面积太小 / 中心挤在一点 = 模型没真在看图)")
    print("=" * 78)
    print(f"  {'类别':<22s}{'n':>5s}{'面积%':>8s}{'中心(cx,cy)':>16s}"
          f"{'跨度(dx,dy)':>16s}{'尺寸mm':>13s}")
    collapsed, tiny, centers = [], [], {}
    for en, g in sorted(geo.items(), key=lambda t: -t[1]["n"]):
        n = g["n"]
        a = 100.0 * g["a"] / n
        cx, cy = g["cx"] / n, g["cy"] / n
        dx, dy = g["x1"] - g["x0"], g["y1"] - g["y0"]
        wmm, hmm = g["wmm"] / n, g["hmm"] / n
        centers[en] = (cx, cy)
        print(f"  {en:<22s}{n:>5d}{a:>8.3f}{f'({cx:.2f},{cy:.2f})':>16s}"
              f"{f'({dx:.2f},{dy:.2f})':>16s}{f'{wmm:.0f}x{hmm:.0f}':>13s}")
        if n >= 5 and dx < 0.05 and dy < 0.05:
            collapsed.append(en)
        if n >= 5 and a < 0.5:
            tiny.append(en)
    print("-" * 78)
    # 跨类 collapse: 不同类别(不同解剖位置)的平均中心却挤在同一点 —— 比单类跨度更狠,
    # 因为单类跨度小可能只是"这一类总在同一位置"(比如 Cardiomegaly 本来就在中下野)。
    # 把各类中心按 0.05×0.05 网格分桶, 看有没有多个类别挤进同一个桶
    buckets = {}
    for en, (cx, cy) in centers.items():
        buckets.setdefault((int(cx / 0.05), int(cy / 0.05)), []).append(en)
    worst = max(buckets.values(), key=len) if buckets else []
    cross = len(worst) >= 3
    if cross:
        bx, by = [k for k, v in buckets.items() if v is worst][0]
        print(f"  ⚠️ 跨类 collapse: {len(worst)} 个类别的平均框中心挤在同一个 "
              f"0.05×0.05 网格里({bx * 5}~{bx * 5 + 5}%, {by * 5}~{by * 5 + 5}%): {worst}")
        print("     解剖位置完全不同的查询给出同一个框 = 模型在背先验, 不是在定位。")
    if collapsed:
        print(f"  ⚠️ 类内中心退化(同一类所有框落在同一点): {collapsed}")
    if cross or collapsed:
        print("     -> 先 cat 对应 cache 的原始响应, 确认坐标口径(0-100)和解析没错, "
              "再下「模型不行」的结论。")
    if tiny:
        print(f"  ⚠️ 平均面积 <0.5% 全图: {len(tiny)} 类(例: {tiny[:5]})")
        print("     IoU 阈值下这种尺寸基本不可能过, mAP 会趋近 0 —— 不是阈值调错。")
    if not (cross or collapsed or tiny):
        if max(g["n"] for g in geo.values()) < 5:
            print("  ⚠️ 每类样本 <5, 上面的数字只能当定性看, 结论等 smoke/full。")
        else:
            print("  ✅ 框有正常的面积和空间分布, 可以进入 mAP 解读。")
    print("=" * 78)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="CheXOne 胸片17类检测(per-class grounding) + 官方指标评估")
    add_client_args(ap)
    ap.set_defaults(model="chexone")
    # 数据
    ap.add_argument("--ann", default="../track8_detection_sized/pretrain_splits/val.json")
    ap.add_argument("--image-root", default="/mnt/data/jinghansun/data/match_yibao/reorg")
    ap.add_argument("--dataset-path", default=None)
    ap.add_argument("--out", default="./answer_chexone")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--limit-mode", default="random", choices=["random", "head"])
    ap.add_argument("--limit-seed", type=int, default=42)
    # CheXOne 特有
    ap.add_argument("--score-mode", default="yesno", choices=["yesno", "verbal", "const"],
                    help="yesno=先问是非题拿P(yes)当分数兼闸门(默认, 推荐); "
                         "verbal=一轮让模型自报 present+confidence+bbox; "
                         "const=不打分, 所有框 conf=1.0(mAP 会失真, 仅作对照)")
    ap.add_argument("--present-thr", type=float, default=0.5,
                    help="yesno 模式下 P(yes) 低于此值就不要框(省请求、压假阳性)")
    ap.add_argument("--reasoning", action="store_true",
                    help="追加 'Please reason step by step' 走 reasoning 模式(慢但论文说更准)")
    ap.add_argument("--add-hint", action="store_true",
                    help="prompt 里带一句该类定义(帮助零样本锚定, 但会偏离训练分布)")
    ap.add_argument("--classes", default=None,
                    help="只跑这几类, 逗号分隔英文 key; 默认全 17 类")
    ap.add_argument("--coord-mode", default="norm100",
                    choices=["norm100", "auto", "norm1000", "norm1", "abs"],
                    help="默认 norm100 —— CheXOne 训练时坐标就归一化到 0-100"
                         "(原生模板的示例 (12,60),(18,87) 即是)。不要动它。")
    ap.add_argument("--short-side", type=int, default=512,
                    help="PNG 先缩到短边 N(保持长宽比)再送模型 —— 对齐 CheXOne 训练时"
                         "VinDr PNG 的预处理。不要动它。")
    # 图像 / 后处理
    ap.add_argument("--max-side", type=int, default=1344,
                    help="short_side=0 时才生效(按长边缩)。默认走 short-side。")
    ap.add_argument("--png-dir", default="./png_chexone",
                    help="CheXOne 专用 PNG 目录 —— 尺寸(短边512)与 Qwen 链路不同, "
                         "不能复用 png_qwen, 否则同一文件名不同尺寸会串味")
    ap.add_argument("--overwrite-png", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--nms-iou", type=float, default=FUSION_NMS_IOU)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--effusion-null", action="store_true")
    # 缓存 / 分片
    ap.add_argument("--cache-dir", default="./cache_chexone")
    ap.add_argument("--reparse-only", action="store_true")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shard", type=int, default=1,
                    help="多卡各起一进程分片; 缓存按 acc+类 命名, 不会互相踩")
    # 评估
    ap.add_argument("--gt-ann", default=None)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--s-ins", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.model = resolve_model(args.model)
    # 图像已经在 image_prep 阶段缩到短边 512 了(2544x3056 -> 512x615 ≈ 315k px)。
    # 这里的上限必须放得下它, 否则 processor 的 smart_resize 会再缩一次
    # (334 token 的预算只有 262k px, 会把 512x615 压回 448x560, 白缩)。
    if "--max-pixels-tokens" not in sys.argv:
        args.max_pixels_tokens = 640
    if "--min-pixels-tokens" not in sys.argv:
        args.min_pixels_tokens = 256

    class_ens = ([c.strip() for c in args.classes.split(",") if c.strip()]
                 if args.classes else [CATID_TO_EN[cid] for cid, _, _ in _cat.CATEGORY_TABLE])
    bad = [e for e in class_ens if e not in EN_TO_CATID]
    if bad:
        print(f"[ERROR] 未知类别 {bad}; 可用: {sorted(EN_TO_CATID)}")
        return

    if args.dry_run:
        print("=" * 70)
        print(f"model={args.model}  score_mode={args.score_mode}  "
              f"reasoning={args.reasoning}")
        print("=" * 70)
        for en in class_ens:
            r = build_prompts(en, args.score_mode, args.reasoning, args.add_hint)
            print(f"\n--- {en}  (phrase='{PHRASE[en]}')")
            for x in (r if isinstance(r, tuple) else (r,)):
                print(x)
        return

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.png_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    if args.ann:
        files = [p for p, _ in list_images_from_ann(args.ann, args.image_root)]
        print(f"[chexone] from ann: {len(files)} images")
    elif args.dataset_path:
        from image_prep import iter_dcm_files
        files = list(iter_dcm_files(args.dataset_path))
    else:
        print("[ERROR] 需要 --ann 或 --dataset-path")
        return
    if args.limit > 0:
        n_all = len(files)
        if args.limit_mode == "random":
            order = list(range(n_all))
            _random.Random(args.limit_seed).shuffle(order)
            files = [files[i] for i in sorted(order[:args.limit])]
        else:
            files = files[:args.limit]
        print(f"[chexone] --limit {args.limit} ({args.limit_mode}) -> "
              f"实跑 {len(files)}/{n_all} 张")
    if args.n_shard > 1:
        files = files[args.shard::args.n_shard]
        print(f"[chexone] shard {args.shard}/{args.n_shard} -> {len(files)} 张")
    if not files:
        print("[ERROR] 未找到 DICOM")
        return

    # ---- 模型 ----
    if args.backend != "local":
        print("[ERROR] CheXOne 只支持 backend=local(本机 transformers)。"
              "vLLM 在服务器上跑不了; 真要用 OpenAI 兼容端点得自己起服务。")
        return
    runner = None
    if not args.reparse_only:
        n_todo = 0
        for p in files:
            from image_prep import accession_from_path
            acc = accession_from_path(p)
            need = [_box_tag(en, args) for en in class_ens]
            if args.score_mode == "yesno":
                need += [_yn_tag(en, args) for en in class_ens]
            if not all(os.path.exists(_cpath(args.cache_dir, acc, t)) for t in need):
                n_todo += 1
        print(f"[chexone] 缓存完全命中 {len(files) - n_todo}/{len(files)}, 需跑 {n_todo} 张")
        if n_todo == 0:
            args.reparse_only = True
        else:
            print(f"[chexone] 加载 {args.model} (min/max vision token "
                  f"{args.min_pixels_tokens}/{args.max_pixels_tokens}) ...")
            runner = QwenLocalRunner(
                model_name=args.model, flash_attn=args.flash_attn,
                min_pixels_tokens=args.min_pixels_tokens,
                max_pixels_tokens=args.max_pixels_tokens,
                torch_dtype=args.dtype,
                gen_config=GenerationConfig(
                    max_new_tokens=(1024 if args.reasoning else 256),
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else 1.0))
            runner.load()
    if args.reparse_only:
        print(f"[chexone] REPARSE-ONLY: 只读 {args.cache_dir}")

    stats, det_count, pres_rate, geo = {}, {}, {}, {}
    n_ok = n_fail = 0
    t_start = time.time()
    for dcm in pbar(files, total=len(files), desc="chexone infer"):
        try:
            t0 = time.time()
            res, acc, present = infer_one(runner, dcm, class_ens, args, stats, geo)
            res["ProcessingTime_ms"] = int((time.time() - t0) * 1000)
            with open(os.path.join(args.out, f"{acc}.json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            n_ok += 1
            for d in res["Prediction"]["Detections"]:
                det_count[d["Finding"]] = det_count.get(d["Finding"], 0) + 1
            for en, p in present.items():
                a, b = pres_rate.get(en, (0.0, 0))
                pres_rate[en] = (a + p, b + 1)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[WARN] {os.path.basename(dcm)}: {type(e).__name__}: {e}")
        if not is_tqdm() and (n_ok + n_fail) % 20 == 0:
            print(f"  processed {n_ok + n_fail}/{len(files)}")

    dt = time.time() - t_start
    print(f"\nDONE. ok={n_ok} fail={n_fail}  用时 {dt:.1f}s "
          f"({dt / max(1, n_ok):.2f}s/img)  -> {args.out}")
    stats["reasoning"] = int(args.reasoning)
    _print_health(stats, max(1, n_ok), len(class_ens))
    _print_geometry(geo)

    print("\n[逐类检出框数 / yes-no 平均分]")
    for en, c in sorted(det_count.items(), key=lambda t: -t[1]):
        pr = pres_rate.get(en)
        extra = f"  P(yes)均值={pr[0] / pr[1]:.3f}" if pr else ""
        print(f"  {en:<22s}{c:>6d}{extra}")
    missing = [e for e in class_ens if det_count.get(e, 0) == 0]
    if missing:
        print(f"  ⚠️ 零检出: {missing}")

    # ---- 评估: 直接调 track8_detection_sized/evaluate.py 的官方规则 ----
    gt_ann = args.gt_ann or args.ann
    if args.no_eval:
        print(f"\n[eval] --no-eval, 跳过。单独出分:"
              f"\n  cd {os.path.relpath(_shared.sized_dir())} && python evaluate.py "
              f"--pred-dir <abs>/{os.path.basename(args.out)} --gt-ann {gt_ann} "
              f"--image-root {args.image_root}")
    elif gt_ann and os.path.exists(gt_ann):
        from evaluate import score_predictions, acc_from_image
        with open(gt_ann, "r", encoding="utf-8") as f:
            gt = json.load(f)
        images = gt.get("images", [])
        # 只评**本次运行**涉及的图。之前用 os.listdir(out) —— 上一次 n 50 / 旧 prompt
        # 留下的 json 会被一起算进来, 换了口径之后分数是两批结果的混合。
        from image_prep import accession_from_path
        done_now = {accession_from_path(p) for p in files}
        on_disk = {os.path.splitext(fn)[0] for fn in os.listdir(args.out)
                   if fn.endswith(".json")}
        done = done_now & on_disk
        stale = on_disk - done_now
        if stale:
            print(f"\n[eval] ⚠️ {args.out} 里另有 {len(stale)} 个不属于本次运行的旧 json, "
                  f"已排除出评估(vis/compare 仍会读到它们, 必要时清目录)")
        n_all = len(images)
        images = [im for im in images if acc_from_image(im) in done]
        if len(images) < n_all:
            print(f"\n[eval] 仅在已产出 answer 的 {len(images)}/{n_all} 张上评估")
        res = score_predictions(args.out, images, args.image_root,
                                no_mm_convert=False, verbose=True)
        if args.s_ins is not None:
            Sins = args.s_ins
            Wins = 0.2 + 0.2 * (1 - Sins)
            Wclin = 1 - Wins
            print(f"\n>>> 总分 = {Sins * Wins * 100 + res['S_clin'] * Wclin * 100:.2f}"
                  f"  (Sins={Sins:.3f} Wins={Wins:.3f} / "
                  f"Sclin={res['S_clin']:.3f} Wclin={Wclin:.3f})")
    else:
        print(f"\n[eval] 跳过: 金标准不存在 {gt_ann}")


if __name__ == "__main__":
    sys.exit(main())
