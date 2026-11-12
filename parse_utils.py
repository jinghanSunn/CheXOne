"""
VLM 原始输出 -> 结构化检测框。这层是整条链路最容易悄悄掉分的地方, 所以写厚一点。

处理的现实问题:
  1. <think>...</think> 思考链前缀(Qwen3.5 thinking 模式)。
  2. ```json ... ``` markdown 围栏。
  3. 模型把数组包进对象: {"detections":[...]} / {"objects":[...]}。
  4. bbox 键名不统一: bbox_2d(Qwen 家族标准) / bbox / box_2d / bounding_box / box。
  5. 坐标尺度三种可能 —— 这是最致命的一条:
       norm1000  : 0-1000 归一化(Qwen3.5 官方约定)
       norm100   : 0-100 归一化(部分数据集/checkpoint 的约定)
       norm1     : 0-1 小数(偶发)
       abs       : 送进去那张图的绝对像素(Qwen2.5-VL 的老约定, 混用模型时会出现)
     auto 模式按坐标幅值判: max>1000 -> abs; max<=1.5 -> norm1; 否则 norm1000。
     ⚠️ auto **无法**区分 norm100 / norm1000(两者都 <=1000), 口径怀疑错了就
        显式写死 --coord-mode norm100 或 norm1000 各跑一遍对比。仅差一个数量级,
        但框面积差 100 倍 —— 症状是"所有框挤在左上角且极小"。
     判错的后果是框整体缩放错位、IoU 全线不过阈值、mAP 看起来"模型完全不行",
     所以 qwen_infer.py 会把每张图判到的模式计数打出来, 异常分布一眼可见。
  6. 坐标顺序颠倒(x2<x1)、越界、退化框(宽或高<=0) -> 修正或丢弃。
  7. label 用中文/复数/带空格/下划线写回来 -> 规范化 + 同义词表映射。
  8. 没有 confidence -> 按输出顺序给递减的伪置信度(并计数告警)。mAP 靠排序,
     全部同分会让 PR 曲线退化。

无外部依赖(仅 re/json)。
"""
import re
import json

from _shared import _cat

EN_TO_CATID = _cat.EN_TO_CATID
CN_TO_CATID = _cat.CN_TO_CATID
CATID_TO_EN = _cat.CATID_TO_EN

# 规范化后的别名 -> 标准英文 key
_ALIAS = {
    "consolidation": "Consolidation", "airspaceopacity": "Consolidation",
    "patchyopacity": "Consolidation", "opacity": "Consolidation",
    "pneumonia": "Consolidation", "infiltrate": "Consolidation",
    "fibrosis": "Fibrosis", "fibroticband": "Fibrosis", "fibroticstrand": "Fibrosis",
    "scarring": "Fibrosis",
    "nodule": "Nodule", "pulmonarynodule": "Nodule", "lungnodule": "Nodule",
    "cardiomegaly": "Cardiomegaly", "enlargedheart": "Cardiomegaly",
    "enlargedcardiacsilhouette": "Cardiomegaly", "heart": "Cardiomegaly",
    "pleuraleffusion": "PleuralEffusion", "effusion": "PleuralEffusion",
    "pleuralfluid": "PleuralEffusion",
    "emphysema": "Emphysema", "hyperinflation": "Emphysema",
    "pleuralthickening": "PleuralThickening", "thickenedpleura": "PleuralThickening",
    "calcification": "Calcification", "calcifiednodule": "Calcification",
    "calcified": "Calcification",
    "infiltration": "Infiltration", "interstitialchange": "Infiltration",
    "interstitial": "Infiltration", "reticularopacity": "Infiltration",
    "mass": "Mass", "lungmass": "Mass", "pulmonarymass": "Mass", "tumor": "Mass",
    "pneumothorax": "Pneumothorax", "ptx": "Pneumothorax",
    "bronchiectasis": "Bronchiectasis",
    "atelectasis": "Atelectasis", "collapse": "Atelectasis", "lungcollapse": "Atelectasis",
    "edema": "Edema", "pulmonaryedema": "Edema", "oedema": "Edema",
    "cavity": "Cavity", "cavitation": "Cavity", "cavitarylesion": "Cavity",
    "mediastinalwidening": "MediastinalWidening", "widenedmediastinum": "MediastinalWidening",
    "mediastinum": "MediastinalWidening",
    "airbronchogram": "AirBronchogram", "airbronchograms": "AirBronchogram",
}
# 标准 key 自身也要能命中
for _en in EN_TO_CATID:
    _ALIAS.setdefault(_en.lower(), _en)

_BBOX_KEYS = ("bbox_2d", "bbox", "box_2d", "box", "bounding_box", "coordinates", "rect")
_LABEL_KEYS = ("label", "class_name", "finding", "category", "name", "class", "ref")
_CONF_KEYS = ("confidence", "score", "conf", "probability", "prob")
_LIST_KEYS = ("detections", "objects", "results", "findings", "boxes", "data", "lesions")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
# CheXOne 的最终答案约定: "推理过程 \boxed{ 答案 }"(论文 §4.2 Format Reward)。
# 与 Qwen3.5 的 <think> 是两套东西, 必须单独处理 —— 不剥的话, reasoning 模式
# 下整段推理文本会被当成答案去 parse, 直接 no_json。
_BOXED_RE = re.compile(r"\\boxed\s*\{")


# --------------------------------------------------------------------------
def strip_wrappers(text):
    """剥 <think> / markdown 围栏, 返回最可能是 JSON 的那段字符串。"""
    if not text:
        return ""
    t = _THINK_RE.sub("", text)
    # 没闭合的 <think>(被 max_tokens 截断): 取最后一个 </think> 之后
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    elif "<think>" in t:
        t = t.rsplit("<think>", 1)[-1]
    m = _FENCE_RE.search(t)
    if m:
        return m.group(1).strip()
    return t.strip()


def _balanced_slice(s, open_ch, close_ch):
    """取第一个 open_ch 起的配平子串(忽略字符串字面量内的括号)。"""
    start = s.find(open_ch)
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def extract_boxed(text, last=True):
    """取 \\boxed{...} 花括号里的内容(CheXOne 最终答案)。没有返回 None。

    取**最后一个** \boxed —— reasoning 模式下模型偶尔会在推理中途写一个试探性的
    \boxed, 最后一个才是它真正提交的答案。
    """
    if not text:
        return None
    idxs = [m.start() for m in _BOXED_RE.finditer(text)]
    if not idxs:
        return None
    for i in (reversed(idxs) if last else idxs):
        seg = _balanced_slice(text[i:], "{", "}")
        if seg:
            return seg[1:-1].strip()
    return None


# Qwen2.5-VL 原生 grounding 标签(CheXOne 训练数据用的就是它):
#   <|ref|>target name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>
# 坐标是**归一化到 0-100**(不是 Qwen 官方 demo 里的 0-1000)—— CheXOne 的
# 训练数据把 VinDr 的 PNG 缩到短边 512 后按 0-100 归一化标注, 示例即
# <|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>。
# 按 0-1000 解释会整体缩小 10 倍、面积小 100 倍。
# 两套标签都要认:
#   <|ref|>..<|/ref|><|box|>..<|/box|>                 CheXOne 原生模板(普通文本, decode 后还在)
#   <|object_ref_start|>..<|object_ref_end|><|box_start|>..<|box_end|>
#                                                     Qwen2.5-VL 原版(**special token**,
#                                                     skip_special_tokens=True 会被剥掉,
#                                                     只剩 "name(x1,y1),(x2,y2)")
# 所以最后还要有一个裸 "(x1,y1),(x2,y2)" 兜底, 否则 special 版本会被静默丢成 no_json。
_NUM = r"(-?\d+(?:\.\d+)?)"
_PAIR2 = (r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)\s*,\s*"
          r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)")
_REF_OPEN = r"<\|\s*(?:ref|object_ref_start)\s*\|>"
_REF_CLOSE = r"<\|\s*(?:/\s*ref|object_ref_end)\s*\|>"
_BOX_OPEN = r"<\|\s*(?:box|box_start)\s*\|>"
_BOX_CLOSE = r"<\|\s*(?:/\s*box|box_end)\s*\|>"
_REFBOX_RE = re.compile(_REF_OPEN + r"\s*(.*?)\s*" + _REF_CLOSE + r"\s*"
                        + _BOX_OPEN + r"\s*" + _PAIR2 + r"\s*" + _BOX_CLOSE, re.DOTALL)
_BOXONLY_RE = re.compile(_BOX_OPEN + r"\s*" + _PAIR2 + r"\s*" + _BOX_CLOSE)
_TUPLE_RE = re.compile(_PAIR2)
# reasoning prompt 里给的示例框。模型在推理正文里复述它不代表预测。
_EXAMPLE_BOX = [12.0, 60.0, 18.0, 87.0]


def _scan_grounding(src):
    hits = [(m.group(1).strip(), [float(m.group(i)) for i in range(2, 6)])
            for m in _REFBOX_RE.finditer(src)]
    if hits:
        return hits, "tag"
    hits = [(None, [float(m.group(i)) for i in range(1, 5)])
            for m in _BOXONLY_RE.finditer(src)]
    if hits:
        return hits, "tag"
    hits = [(None, [float(m.group(i)) for i in range(1, 5)])
            for m in _TUPLE_RE.finditer(src)]
    if hits:
        return hits, "tuple"
    return [], None


def extract_grounding(text, stats=None):
    """抽出原生 grounding 标注 -> [(label|None, [x1,y1,x2,y2]), ...]。

    取框规则(按优先级):
      1. 有 \\boxed 且里面有框 -> 只信 \\boxed 里的(reasoning 模式的最终答案)。
      2. 有 \\boxed 但里面没框 -> 全文只取**最后一个**框。推理正文里前面的框
         是中间草稿, 全收会把一个 query 变成多个框(堆叠的来源之一)。
      3. 没有 \\boxed(instruct 模式) -> 全文所有框都收(模型可能吐多个实例)。
    非 \\boxed 来源里与 prompt 示例完全相同的框视为复述, 丢弃。
    """
    def bump(k):
        if stats is not None:
            stats[k] = stats.get(k, 0) + 1

    if not text:
        return []
    boxed = extract_boxed(text)
    if boxed:
        hits, kind = _scan_grounding(boxed)
        if hits:
            bump(f"gsrc_boxed_{kind}")
            return hits
    hits, kind = _scan_grounding(text)
    n0 = len(hits)
    hits = [h for h in hits if h[1] != _EXAMPLE_BOX]
    if len(hits) < n0:
        bump("example_echo")
    if not hits:
        return []
    if boxed is not None:
        hits = hits[-1:]
        bump("gsrc_text_last")
    else:
        bump(f"gsrc_text_{kind}")
    return hits


def extract_json(text, prefer_boxed=True):
    """从自由文本里抠出 JSON(数组优先, 其次对象)。失败返回 None。

    prefer_boxed: 文本里有 \\boxed{...} 时优先解析框里的内容。Qwen3.5 从不输出
    \\boxed, 所以这个开关对原链路是 no-op, 可以安全默认开。
    """
    s = strip_wrappers(text)
    if not s:
        return None
    if prefer_boxed:
        # 先在原始文本里找(推理正文里可能带 ```json``` 干扰), 再在剥壳后的文本里找
        for src in (text, s):
            boxed = extract_boxed(src)
            if boxed:
                try:
                    return json.loads(boxed)
                except Exception:
                    pass
                cand = _balanced_slice(boxed, "[", "]") or _balanced_slice(boxed, "{", "}")
                if cand:
                    try:
                        return json.loads(re.sub(r",\s*([\]}])", r"\1", cand))
                    except Exception:
                        pass
    for direct in (s,):
        try:
            return json.loads(direct)
        except Exception:
            pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        cand = _balanced_slice(s, open_ch, close_ch)
        if cand:
            try:
                return json.loads(cand)
            except Exception:
                # 常见小毛病: 尾逗号 / 单引号
                fixed = re.sub(r",\s*([\]}])", r"\1", cand)
                try:
                    return json.loads(fixed)
                except Exception:
                    continue
    return None


def _as_item_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in _LIST_KEYS:
            if isinstance(obj.get(k), list):
                return obj[k]
        # {"Nodule": [[..]], ...} 这种类->框映射
        out = []
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                for b in v:
                    out.append({"bbox_2d": b, "label": k})
            elif isinstance(v, list) and len(v) == 4 and all(
                    isinstance(x, (int, float)) for x in v):
                out.append({"bbox_2d": v, "label": k})
        if out:
            return out
        # 单个检测对象
        if any(k in obj for k in _BBOX_KEYS):
            return [obj]
    return []


def normalize_label(raw):
    """任意写法 -> 标准英文 key; 认不出返回 None。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in CN_TO_CATID:                     # 中文原名
        return CATID_TO_EN[CN_TO_CATID[s]]
    key = re.sub(r"[^a-z0-9]", "", s.lower())
    if key in _ALIAS:
        return _ALIAS[key]
    if key.endswith("s") and key[:-1] in _ALIAS:   # 简单复数
        return _ALIAS[key[:-1]]
    # 中文子串兜底(模型写 "肺结节影")
    for cn, cid in CN_TO_CATID.items():
        if cn and cn in s:
            return CATID_TO_EN[cid]
    return None


def _pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    # 大小写不敏感兜底
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in low and low[k] is not None:
            return low[k]
    return None


def _coerce_box(raw):
    """-> [x1,y1,x2,y2] float, 失败 None。支持 dict 形式 {xmin,ymin,xmax,ymax}。"""
    if isinstance(raw, dict):
        got = []
        for ks in (("x1", "xmin", "left"), ("y1", "ymin", "top"),
                   ("x2", "xmax", "right"), ("y2", "ymax", "bottom")):
            v = _pick(raw, ks)
            if v is None:
                return None
            got.append(v)
        raw = got
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]
    except (TypeError, ValueError):
        return None


def detect_coord_mode(boxes, declared="auto"):
    """boxes: list of [x1,y1,x2,y2] 原始值。返回 'norm1000'|'norm1'|'abs'。"""
    if declared != "auto":
        return declared
    if not boxes:
        return "norm1000"
    mx = max(max(abs(v) for v in b) for b in boxes)
    if mx > 1000.5:
        return "abs"
    if mx <= 1.5:
        return "norm1"
    return "norm1000"


def to_pixel(box, mode, W_in, H_in, W0, H0):
    """原始坐标 -> **原图像素** xyxy。

    W_in/H_in = 实际送进模型那张图的宽高(abs 模式下坐标相对于它);
    W0/H0     = 原始 DICOM 的宽高(输出要落回这个坐标系)。
    """
    x1, y1, x2, y2 = box
    # 归一化口径下坐标越界(留 2% 容差) = 口径不符或模型乱吐, 直接判坏框。
    # 不能 clamp: 450 按 0-100 解释 clamp 到右边缘后会变成 1px 宽的"合法"框,
    # 带着高分混进 AP 计算, 而 bad_box 计数里看不到它。
    lim = {"norm1000": 1000.0, "norm100": 100.0, "norm1": 1.0}.get(mode)
    if lim is not None and max(x1, y1, x2, y2) > lim * 1.02:
        return None
    if mode == "norm1000":
        sx, sy = W0 / 1000.0, H0 / 1000.0
    elif mode == "norm100":
        # 有些 checkpoint/数据集用 0-100 口径。看起来只差一个数量级, 但解释错了
        # 框会缩小 10 倍 -> 面积小 100 倍, mAP 直接归零且看不出原因。
        sx, sy = W0 / 100.0, H0 / 100.0
    elif mode == "norm1":
        sx, sy = float(W0), float(H0)
    else:  # abs, 相对送入图
        sx, sy = W0 / max(W_in, 1), H0 / max(H_in, 1)
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, W0 - 1.0)); x2 = max(0.0, min(x2, float(W0)))
    y1 = max(0.0, min(y1, H0 - 1.0)); y2 = max(0.0, min(y2, float(H0)))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def parse_detections(text, W_in, H_in, W0, H0, coord_mode="auto",
                     allow_catids=None, stats=None, default_label=None):
    """VLM 文本 -> [{box_px, en, cid, conf, max_diameter_ratio, effusion_grade}]。

    default_label: 单类查询(CheXOne abnormality grounding 是"给异常名 -> 给一个框")
    下模型不会回写 label —— 框本身就是答案。这时把查询的类别填进去, 否则会被
    bad_label 全丢。

    stats: 可选 dict, 原地累加 {'no_json','bad_label','bad_box','no_conf','mode_<m>'}。
    """
    def bump(k):
        if stats is not None:
            stats[k] = stats.get(k, 0) + 1

    # 原生 grounding 标签优先 —— JSON 路径认不出 <|ref|>/<|box|>, 必须由这里接住
    ghits = extract_grounding(text, stats=stats)
    if ghits:
        bump("grounding_hit")
        items = []
        for lab, b in ghits:
            en = normalize_label(lab) if lab else None
            if default_label is not None:
                # 单类查询: 类别由**问题**决定, 不由 <|ref|> 里的文字决定。
                # 否则问 Pneumothorax 时模型写 ref="opacity" 会被 _ALIAS 映射成
                # Consolidation, 而分数却是 Pneumothorax 的 P(yes) —— 跨类串标。
                if lab and en != default_label:
                    bump("ref_mismatch")
                en = default_label
            items.append({"bbox_2d": b, "label": en})
    else:
        obj = extract_json(text)
        if obj is None:
            bump("no_json")
            return []
        # 裸 [x1,y1,x2,y2](CheXOne 可能的输出形态) -> 补成带 label 的一项
        if (isinstance(obj, (list, tuple)) and len(obj) == 4
                and all(isinstance(v, (int, float)) for v in obj)):
            obj = [{"bbox_2d": list(obj), "label": default_label}]
        items = _as_item_list(obj)
    if not items:
        bump("empty")
        return []

    raw_boxes, staged = [], []
    for it in items:
        if not isinstance(it, dict):
            # 可能是裸 [x1,y1,x2,y2]
            b = _coerce_box(it)
            if b is None:
                bump("bad_box")
                continue
            staged.append((b, None, None, None, None))
            raw_boxes.append(b)
            continue
        b = _coerce_box(_pick(it, _BBOX_KEYS))
        if b is None:
            bump("bad_box")
            continue
        en = normalize_label(_pick(it, _LABEL_KEYS))
        conf = _pick(it, _CONF_KEYS)
        mdr = _pick(it, ("max_diameter_ratio", "maxdiameterratio", "max_diameter"))
        eg = _pick(it, ("effusion_grade", "effusiongrade", "grade"))
        staged.append((b, en, conf, mdr, eg))
        raw_boxes.append(b)

    mode = detect_coord_mode(raw_boxes, coord_mode)
    bump(f"mode_{mode}")

    out = []
    n_noconf = 0
    for b, en, conf, mdr, eg in staged:
        if en is None:
            if default_label is not None:      # 单类查询: 框就是这一类
                en = default_label
            else:
                bump("bad_label")
                continue
        cid = EN_TO_CATID.get(en)
        if cid is None:
            bump("bad_label")
            continue
        if allow_catids is not None and cid not in allow_catids:
            bump("out_of_group")
            continue
        px = to_pixel(b, mode, W_in, H_in, W0, H0)
        if px is None:
            bump("bad_box")
            continue
        try:
            c = float(conf)
            if not (0.0 <= c <= 1.0):
                c = max(0.0, min(1.0, c / 100.0 if c > 1.0 else c))
        except (TypeError, ValueError):
            c = None
            n_noconf += 1
        try:
            mdr = float(mdr) if mdr is not None else None
        except (TypeError, ValueError):
            mdr = None
        try:
            eg = int(eg) if eg is not None else None
            if eg not in (1, 2, 3):
                eg = None
        except (TypeError, ValueError):
            eg = None
        out.append({"box_px": px, "en": en, "cid": cid, "conf": c,
                    "max_diameter_ratio": mdr, "effusion_grade": eg})

    # 伪置信度: 按输出顺序递减(模型通常先写最确信的)。mAP 需要可排序的分数。
    if n_noconf:
        bump("no_conf")
        n = len(out)
        for i, d in enumerate(out):
            if d["conf"] is None:
                d["conf"] = round(max(0.05, 0.95 - 0.05 * i), 4)
    return out


if __name__ == "__main__":
    demo = """<think>Let me look...</think>
```json
[{"bbox_2d":[120,330,260,470],"label":"肺结节","confidence":0.72},
 {"bbox_2d":[600,700,980,900],"label":"pleural effusions","effusion_grade":2}]
```"""
    st = {}
    for d in parse_detections(demo, 1024, 1024, 2048, 2560, stats=st):
        print(d)
    print("stats:", st)
