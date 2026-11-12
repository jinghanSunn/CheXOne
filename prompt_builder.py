"""
Qwen3.5 胸片 17 类病灶 grounding prompt 构造。

设计要点(踩坑预判, 不是随手写的 prompt):
  1. Qwen3.5 的 grounding 输出约定是 JSON 数组, 每项 {"bbox_2d":[x1,y1,x2,y2], "label":...},
     坐标为 **0-1000 归一化**。prompt 里把这个格式**显式写死**, 不靠模型自己猜 ——
     FindIt benchmark 的结论是: 不约束格式时 Qwen 系列虽然 mIoU 还行, 但 label↔box
     错位会让 F1 直接塌到 0。所以必须强制 JSON、强制每框自带 label。
  2. label 用【英文 key】而不是中文, 避免中文同义词(片状模糊影/实变影)导致映射失败;
     但 prompt 里给出中英对照表帮模型锚定语义。
  3. 显式要求 "confidence": 0~1 —— mAP 是按置信度排序算 PR 曲线的, 没有分数的检测
     结果 AP 会严重失真(所有框等权 = 随机排序)。模型给的分数虽不校准, 但单调性有用。
  4. 允许输出空数组(正常片), 并明确"宁可不报也不要编" —— 抑制 VLM 的幻觉式铺框。
  5. --prompt-mode group: 按 categories.CLASS_GROUPS 分 3 轮问(每轮 5-6 类),
     类别表更短 -> 指令遵循更稳, 代价是 3 倍请求量。

无外部依赖。
"""
from _shared import _cat

CATEGORY_TABLE = _cat.CATEGORY_TABLE
CLASS_GROUPS = _cat.CLASS_GROUPS
CATID_TO_EN = _cat.CATID_TO_EN
CATID_TO_CN = _cat.CATID_TO_CN

# 给模型的每类一句话定义(帮助零样本锚定, 不写定义时模型容易把纤维条索/间质改变混为一谈)
CLASS_HINT = {
    "Consolidation":       "patchy/airspace opacity, lobar or segmental dense shadow",
    "Fibrosis":            "linear/cord-like strands, usually apical or peripheral",
    "Nodule":              "round focal opacity, diameter <= 30 mm",
    "Cardiomegaly":        "enlarged cardiac silhouette (CTR > 0.5); box the heart",
    "PleuralEffusion":     "blunted costophrenic angle / meniscus fluid at lung base",
    "Emphysema":           "hyperlucent over-inflated lung zone, flattened diaphragm",
    "PleuralThickening":   "thickened pleural line along the chest wall or apex",
    "Calcification":       "very dense (bone-white) small focus inside the lung",
    "Infiltration":        "diffuse reticular/interstitial change, hazy texture",
    "Mass":                "round or lobulated opacity, diameter > 30 mm",
    "Pneumothorax":        "visible visceral pleural line with no lung markings beyond it",
    "Bronchiectasis":      "dilated thick-walled bronchi, tram-track or ring shadows",
    "Atelectasis":         "volume loss, displaced fissure, collapsed lung segment",
    "Edema":               "bilateral perihilar bat-wing haziness, Kerley lines",
    "Cavity":              "lucent cavity with a definable wall inside a consolidation",
    "MediastinalWidening": "widened mediastinal contour; box the mediastinum",
    "AirBronchogram":      "air-filled bronchi outlined by surrounding consolidation",
}

_SYSTEM = (
    "You are a senior thoracic radiologist performing lesion detection on a "
    "frontal chest radiograph (CXR). You localize findings precisely and you "
    "never invent findings that are not visible."
)

_FORMAT_BLOCK = """\
OUTPUT FORMAT (strict, nothing else):
Return a single JSON array. Each element MUST be exactly:
  {"bbox_2d": [x1, y1, x2, y2], "label": "<EnglishKey>", "confidence": <float 0~1>}
- Coordinates are normalized to the range 0-1000 over the WHOLE image
  (x1,y1 = top-left; x2,y2 = bottom-right; x1<x2, y1<y2).
- "label" MUST be copied verbatim from the allowed list above. No synonyms, no Chinese.
- "confidence" is your own certainty for THAT box, used for ranking. Spread the
  values out; do not give every box the same number.
- One element per lesion. If the same finding appears bilaterally, emit TWO boxes.
- If you see nothing abnormal, return exactly: []
- Output ONLY the JSON array. No markdown fence, no explanation, no prose."""

_EXTRA_ATTR = """\
EXTRA ATTRIBUTES (only when applicable):
- For a "Mass" box, also add "max_diameter_ratio": <float 0~1>, the lesion's longest
  axis as a fraction of the image width.
- For a "PleuralEffusion" box, also add "effusion_grade": 1 | 2 | 3
  (1 = small, fluid below the 4th anterior rib; 2 = moderate;
   3 = large, fluid level above the 2nd anterior rib)."""


def _class_lines(catids):
    lines = []
    for cid in sorted(catids):
        en = CATID_TO_EN[cid]
        cn = CATID_TO_CN[cid]
        hint = CLASS_HINT.get(en, "")
        lines.append(f"  - {en}  ({cn}) : {hint}")
    return "\n".join(lines)


def build_prompt(catids=None, want_attrs=True, extra_instruction=None):
    """构造一轮 grounding 的 user prompt。catids=None 表示全 17 类。"""
    if catids is None:
        catids = [cid for cid, _, _ in CATEGORY_TABLE]
    head = (
        f"Detect and localize ALL abnormal findings in this chest X-ray.\n"
        f"You may ONLY use these {len(catids)} finding labels:\n"
        f"{_class_lines(catids)}\n\n"
    )
    body = _FORMAT_BLOCK
    if want_attrs:
        has_mass = _cat.EN_TO_CATID["Mass"] in catids
        has_eff = _cat.EN_TO_CATID["PleuralEffusion"] in catids
        if has_mass or has_eff:
            body += "\n\n" + _EXTRA_ATTR
    if extra_instruction:
        body += "\n\n" + extra_instruction.strip()
    return head + body


def build_rounds(prompt_mode="all", want_attrs=True, extra_instruction=None):
    """返回 [(round_name, prompt_text, catid_whitelist_set), ...]。

    prompt_mode:
      all     -> 1 轮问全 17 类(请求少, 省 token; 类别表长, 指令遵循稍弱)
      group   -> 3 轮, 按 categories.CLASS_GROUPS 分组(focal_small/diffuse_large/
                 pleural_struct), 每轮类别表短 -> 更稳, 代价 3x 请求量。
                 白名单过滤与 inference_sized.py 的融合口径一致: 本轮只收本组类的框。
      perclass-> 17 轮, 每轮只问 1 类(白名单最短, 指令遵循最稳), 代价 17x 请求量。
                 注意: 单类询问下模型倾向"有", 假阳性会上升 -> precision 可能跌,
                 配 --strict-group(越类丢弃) + 评测时看 precision 而非只看 recall。
    """
    if prompt_mode == "all":
        allids = {cid for cid, _, _ in CATEGORY_TABLE}
        return [("all", build_prompt(None, want_attrs, extra_instruction), allids)]
    if prompt_mode == "group":
        rounds = []
        for g in _cat.GROUP_NAMES:
            ids = set(CLASS_GROUPS[g])
            rounds.append((g, build_prompt(sorted(ids), want_attrs, extra_instruction), ids))
        return rounds
    if prompt_mode == "perclass":
        rounds = []
        for cid, _cn, en in CATEGORY_TABLE:
            rounds.append((en, build_prompt([cid], want_attrs, extra_instruction), {cid}))
        return rounds
    raise ValueError(f"unknown prompt_mode: {prompt_mode} (expect all|group|perclass)")


def system_prompt():
    return _SYSTEM


# ==========================================================================
# 区域分类(给定 GT 框, 问"框里是什么") —— 把【定位】和【识别】解耦的消融。
# 检测 mAP 低有两种可能: 找不到病灶, 或找到了但叫错名字。喂 GT 框能单独量后者,
# 得到的是这条链路的【识别上限】: 连框都给了还认不出, 说明瓶颈在识别而非定位;
# 反之识别很准而检测 mAP 低, 瓶颈就是定位/框的标注习惯。
# ==========================================================================
_SYSTEM_REGION = (
    "You are a senior thoracic radiologist. You are shown a chest radiograph "
    "and one specific region of interest that has been confirmed abnormal by "
    "a radiologist. Your job is to name which finding it is. The region IS "
    "abnormal - do not answer that it is normal."
)

_REGION_FORMAT = """\
OUTPUT FORMAT (strict, nothing else):
Return a single JSON object, exactly:
  {"label": "<EnglishKey>", "top3": ["<1st>", "<2nd>", "<3rd>"], "confidence": <float 0~1>}
- "label" and every entry of "top3" MUST be copied verbatim from the list above.
  No synonyms, no Chinese, no "Normal", no "No Finding".
- "top3" is your ranked differential, most likely first; "label" must equal top3[0].
- Output ONLY the JSON object. No markdown fence, no explanation."""

_REGION_INTRO = {
    "coord": (
        "The region of interest is the axis-aligned box [{x1}, {y1}, {x2}, {y2}] "
        "in coordinates normalized to 0-1000 over the whole image "
        "(x1,y1 = top-left; x2,y2 = bottom-right).\n"
        "Look ONLY at that box and name the finding inside it."),
    "marked": (
        "The region of interest is outlined by the bright rectangle drawn on the image.\n"
        "Look ONLY inside that rectangle and name the finding inside it."),
    "crop": (
        "The image shown is a crop centred on the region of interest "
        "(some surrounding lung is kept for context).\n"
        "Name the finding located in the centre of this crop."),
}


def build_region_prompt(mode, box_norm1000=None, catids=None):
    """区域分类 prompt。mode: coord | marked | crop。
    box_norm1000: (x1,y1,x2,y2) 0-1000 整数, 仅 coord 模式需要。"""
    if mode not in _REGION_INTRO:
        raise ValueError(f"unknown region mode: {mode} (expect coord|marked|crop)")
    if catids is None:
        catids = [cid for cid, _, _ in CATEGORY_TABLE]
    intro = _REGION_INTRO[mode]
    if mode == "coord":
        if not box_norm1000:
            raise ValueError("coord 模式必须给 box_norm1000")
        x1, y1, x2, y2 = [int(round(v)) for v in box_norm1000]
        intro = intro.format(x1=x1, y1=y1, x2=x2, y2=y2)
    return (
        f"{intro}\n\n"
        f"It must be exactly one of these {len(catids)} findings:\n"
        f"{_class_lines(catids)}\n\n"
        f"{_REGION_FORMAT}"
    )


def region_system_prompt():
    return _SYSTEM_REGION


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, p, ids in build_rounds(mode):
        print("=" * 70)
        print(f"[round={name}]  {len(ids)} classes")
        print("=" * 70)
        print(p)
        print()
