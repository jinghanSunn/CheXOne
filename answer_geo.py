"""CheXOne answer 框几何诊断 —— 回答"是缩放问题还是塌缩"。

用法:
  # 1) 单张 answer 反解: norm1000 / 物理尺寸 / 面积% / 中心
  python answer_geo.py answer_chexone/JSDR202009110764.json

  # 2) 全目录按类几何(不需要 GT —— 只看框自己落在哪、多大、散不散):
  python answer_geo.py answer_chexone

  # 3) 全目录 vs GT 轴分解(判断 x/y 哪根轴坏、能不能用缩放修):
  python answer_geo.py answer_chexone \
      --gt-ann ../track8_detection_sized/pretrain_splits/val.json

mode 2 的核心用途: 区分「退化」和「定位不准」—— 两者在 mAP 上都是 0,
  但判别方法不同:
    框**收敛**到同一个点(类内跨度 ~0 且类间中心也重合) -> 退化, 模型没在看图
    框**分散**在目标附近            -> 定位不准, 是真在做估计
  再看 Cardiomegaly 这类**大结构**: 心影在 448x560 输入下依然清晰可见,
  它要是也塌, 就是纯退化而非"小病灶看不见"。

关键判据(mode 2):
  iou_x = 只看 x 区间的一维 IoU; iou_y 同理。
  **IoU(y 修成 GT) == iou_x**, **IoU(x 修成 GT) == iou_y** —— 所以:
    iou_x 高、iou_y 低 -> 只有 y 坏, x 定位是好的(有救)
    两者都低          -> 整框崩, 不是某根轴的问题
  再看 pred_h/gt_h 的中位数和离散度:
    集中在某个常数 c  -> y 是**固定缩放偏差**, 可以修(改换算或后处理)
    离散且都 <0.1     -> y 是**塌缩**(恒输出小框), 缩放修不了
"""
import json
import os
import sys
import statistics as st

# ---------------------------------------------------------------- utils
def _med(xs):
    return st.median(xs) if xs else float("nan")


def _iqr(xs):
    if len(xs) < 4:
        return float("nan")
    xs = sorted(xs)
    return xs[3 * len(xs) // 4] - xs[len(xs) // 4]


def _iou_1d(a1, a2, b1, b2):
    """区间 [a1,a2] 与 [b1,b2] 的一维 IoU。"""
    inter = min(a2, b2) - max(a1, b1)
    if inter <= 0:
        return 0.0
    union = max(a2, b2) - min(a1, b1)
    return inter / union if union > 0 else 0.0


def _iou(a, b):
    """轴对齐盒: 二维 IoU == iou_x * iou_y。"""
    return _iou_1d(a[0], a[2], b[0], b[2]) * _iou_1d(a[1], a[3], b[1], b[3])


# ---------------------------------------------------------------- mode 1
def single(path):
    d = json.load(open(path, encoding="utf-8"))
    P = d["Prediction"]
    W, H = P["ImageSize"]["width_px"], P["ImageSize"]["height_px"]
    ps = P["ImageSize"]["pixel_spacing_mm"][0]
    img_mm = (W * ps) * (H * ps)
    print("image %dx%d px  ps=%s  ->  %.0f x %.0f mm  全图面积 %.0f mm2"
          % (W, H, ps, W * ps, H * ps, img_mm))
    hdr = ("%-22s%7s%22s%12s%10s%9s%16s%8s"
           % ("Finding", "Conf", "norm1000", "尺寸mm", "面积mm2", "面积%",
              "中心(cx,cy)", "侧别"))
    print(hdr); print("-" * len(hdr))
    for det in P["Detections"]:
        x1, y1, x2, y2 = det["BBox_mm"]
        px = [v / ps for v in (x1, y1, x2, y2)]
        n = [round(px[0] / W * 1000), round(px[1] / H * 1000),
             round(px[2] / W * 1000), round(px[3] / H * 1000)]
        a = (x2 - x1) * (y2 - y1)
        cx, cy = (x1 + x2) / 2 / (W * ps), (y1 + y2) / 2 / (H * ps)
        lat = max(det["Laterality_probabilities"],
                  key=det["Laterality_probabilities"].get)
        print("%-22s%7.3f%22s%12s%10.1f%9.4f%16s%8s"
              % (det["Finding"], det["Confidence"], str(n),
                 "%.1fx%.1f" % (x2 - x1, y2 - y1), a, 100 * a / img_mm,
                 "(%.3f,%.3f)" % (cx, cy), lat))


# ---------------------------------------------------------------- mode 2
def class_geo(ans_dir, scale=1.0):
    """不需要 GT: 只看模型自己吐的框落在哪、多大、散不散。

    scale: 坐标口径重解释系数。**零成本验证口径假说**用 ——
      模型吐的值是 0-100 口径却被按 0-1000 解释时, 框会整体缩小 10 倍
      (面积小 100 倍), 症状就是"所有框挤在左上角且只有几毫米"。
      scale=10 等价于把它们按 0-100 重新解释, 不用重跑模型。
      若重解释后框变合理 -> 口径错了; 若大面积越界 -> 原口径是对的。

    判据(区分「退化」vs「定位不准」, 两者 mAP 都是 0 但性质完全不同):
      类内跨度 ~0 且类间中心重合 -> 退化(模型在输出一个安全解, 没在看图)
      类内跨度大               -> 真在做位置估计, 只是不准
    另外单独盯 y: 已观察到的现象是 y 比 x 塌得更彻底。
    """
    per, n_img, n_box = {}, 0, 0
    n_oob = 0     # 重解释后越界的框数 —— 口径假说的反证
    ally1, ally2, allx1, allx2 = [], [], [], []
    for fn in sorted(os.listdir(ans_dir)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(ans_dir, fn), encoding="utf-8"))
        P = d.get("Prediction", {})
        size = P.get("ImageSize", {})
        W, H = size.get("width_px", 0), size.get("height_px", 0)
        sp = size.get("pixel_spacing_mm") or [1.0, 1.0]
        spc = sp[0] or 1.0
        spr = (sp[1] if len(sp) > 1 and sp[1] else spc)
        if not W or not H:
            continue
        n_img += 1
        for det in P.get("Detections", []):
            en = det.get("Finding")
            bb = det.get("BBox_mm")
            if not en or not bb:
                continue
            x1, y1 = bb[0] / spc, bb[1] / spr
            x2, y2 = bb[2] / spc, bb[3] / spr
            n_box += 1
            if scale != 1.0:
                # 按另一种口径重解释: 归一化坐标 x scale, 再落回像素。
                # 越界的直接 clamp, 并计数 —— 越界多说明原口径才是对的。
                cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
                w2, h2 = (x2 - x1) * scale, (y2 - y1) * scale
                if cx0 / W * scale > 1.0 or cy0 / H * scale > 1.0:
                    n_oob += 1
                x1 = max(0.0, min(cx0 * scale - w2 / 2, W))
                x2 = max(0.0, min(cx0 * scale + w2 / 2, W))
                y1 = max(0.0, min(cy0 * scale - h2 / 2, H))
                y2 = max(0.0, min(cy0 * scale + h2 / 2, H))
            r = per.setdefault(en, dict(n=0, a=[], w=[], h=[], cx=[], cy=[],
                                        x1=[], y1=[], x2=[], y2=[]))
            r["n"] += 1
            r["a"].append(100.0 * (x2 - x1) * (y2 - y1) / (W * H))
            r["w"].append((x2 - x1) * spc)
            r["h"].append((y2 - y1) * spr)
            r["cx"].append((x1 + x2) / 2 / W)
            r["cy"].append((y1 + y2) / 2 / H)
            r["x1"].append(x1 / W); r["x2"].append(x2 / W)
            r["y1"].append(y1 / H); r["y2"].append(y2 / H)
            ally1.append(y1 / H); ally2.append(y2 / H)
            allx1.append(x1 / W); allx2.append(x2 / W)

    print("[geo] %s: %d 张图, %d 个框, %d 类%s"
          % (ans_dir, n_img, n_box, len(per),
             "" if scale == 1.0 else "   <<< 坐标按 x%.0f 重解释" % scale))
    if scale != 1.0 and n_box:
        print("      重解释后越界的框: %d/%d (%.0f%%) —— 比例高说明原口径才是对的"
              % (n_oob, n_box, 100.0 * n_oob / n_box))
    if not per:
        print("[geo] 没有框 —— 要么全被 PRESENT_THR 挡了, 要么目录不对。")
        return
    hdr = ("%-22s%5s%9s%12s%16s%18s%18s"
           % ("类别", "n", "面积%", "尺寸mm", "中心(cx,cy)",
              "中心跨度(dx,dy)", "y区间"))
    print(); print(hdr); print("-" * len(hdr))
    for en in sorted(per, key=lambda e: -per[e]["n"]):
        r = per[en]
        dx = max(r["cx"]) - min(r["cx"])
        dy = max(r["cy"]) - min(r["cy"])
        yy = r["y1"] + r["y2"]
        print("%-22s%5d%9.3f%12s%16s%18s%18s"
              % (en, r["n"], _med(r["a"]),
                 "%.0fx%.0f" % (_med(r["w"]), _med(r["h"])),
                 "(%.3f,%.3f)" % (_med(r["cx"]), _med(r["cy"])),
                 "(%.3f,%.3f)" % (dx, dy),
                 "%.3f~%.3f" % (min(yy), max(yy))))

    # ---- 判读 1: 全局 collapse
    centers = {en: (_med(r["cx"]), _med(r["cy"])) for en, r in per.items()}
    buckets = {}
    for en, (cx, cy) in centers.items():
        buckets.setdefault((int(cx * 20), int(cy * 20)), []).append(en)
    cxs = [c[0] for c in centers.values()]
    cys = [c[1] for c in centers.values()]
    gdx, gdy = max(cxs) - min(cxs), max(cys) - min(cys)
    big = [en for b, ens in buckets.items() if len(ens) >= 3
           for en in ens]

    print("\n  ===== 判读 =====")
    print("  各类中心全局跨度: dx=%.3f  dy=%.3f  (跨 %d 个类)" % (gdx, gdy, len(centers)))
    if big:
        print("  ⚠️ **跨类 collapse**: %d 个类的平均框中心落在同一个 "
              "0.05x0.05 格子里: %s" % (len(big), sorted(big)[:8]))
        print("     解剖位置毫不相干的查询给出同一个框 = 模型在输出一个安全解, "
              "不是在定位。")
    elif gdx < 0.10 and gdy < 0.10:
        print("  ⚠️ 所有类的中心挤在 %.2f 范围内 —— 接近全局退化。" % max(gdx, gdy))
    else:
        print("  ✅ 各类中心是分开的, 说明输出至少与类别相关(不是完全退化的单点)。")

    # ---- 判读 2: y 轴是不是被锁死
    if ally1:
        p90_y2 = sorted(ally2)[int(0.9 * (len(ally2) - 1))]
        mx2, my2 = max(allx2), max(ally2)
        print("\n  全部框: y2 p90=%.3f (max %.3f)   x2 max=%.3f" % (p90_y2, my2, mx2))
        if p90_y2 < 0.10:
            print("  ⚠️ **y 轴锁死在顶部 10%** —— 90% 的框下边界都没超过图的 10%。")
            if mx2 > 2 * my2 + 0.05:
                print("     但 x 能到 %.2f —— x 还残留弱语义先验(左/右肺、纵膈居中), "
                      "y 已完全没信号。" % mx2)
            else:
                print("     x 也一样锁死(max x2=%.3f) —— 两根轴都没信号, 是纯退化。" % mx2)

    # ---- 判读 3: 大结构类是不是也塌(这是分辨率 vs 纯退化的判决点)
    bigcls = [e for e in ("Cardiomegaly", "PleuralEffusion", "Atelectasis",
                          "MediastinalWidening", "Consolidation") if e in per]
    if bigcls:
        print("\n  大结构类(448x560 输入下依然肉眼可辨 —— 分辨率假说的判决点):")
        usable = []
        for en in bigcls:
            r = per[en]
            dx = max(r["cx"]) - min(r["cx"])
            dy = max(r["cy"]) - min(r["cy"])
            print("    %-22s n=%-4d 面积%%=%7.3f  中心=(%.2f,%.2f)  跨度=(%.2f,%.2f)%s"
                  % (en, r["n"], _med(r["a"]), _med(r["cx"]), _med(r["cy"]),
                     dx, dy, "" if r["n"] >= 3 else "   <- n<3, 不构成判决"))
            if r["n"] >= 3:
                usable.append((en, _med(r["a"]), dx, dy,
                               _med(r["cx"]), _med(r["cy"])))
        if not usable:
            print("    (样本不足 —— 需要 >=3 个框才能判, 先多跑几张)")
        else:
            # 判决不看跨度(同类病灶在不同病人间位置本来就集中), 看两个更硬的条件:
            #   1) 面积够大 —— 说明模型画的是"结构"不是"退化小框"
            #   2) 中心不在左上角那个塌缩点附近 —— 说明位置是真估出来的
            ok = [e for e, a, dx, dy, cx, cy in usable
                  if a > 3.0 and (0.15 < cx < 0.85 or cy > 0.15)]
            if ok:
                print("    -> %s 面积/跨度正常: **不是纯退化**, 主因是分辨率下小病灶不可见。"
                      % ", ".join(ok))
                print("       下一步: CX_MAX_PIXELS_TOKENS=1280 重跑同一批对照。")
            else:
                print("    -> 大结构类也塌在角落: **纯退化**, 提高分辨率大概率救不回来。")
                print("       模型在本赛题影像分布上不 work, 建议止损。")


# ---------------------------------------------------------------- mode 3
def load_gt(gt_ann):
    """{acc: [(x1,y1,x2,y2,cid), ...]} 原图像素。与 visualize_detections 同口径。"""
    data = json.load(open(gt_ann, encoding="utf-8"))
    out = {}
    for im in data.get("images", []):
        acc = im.get("accession_number") or im.get("acc") \
            or os.path.splitext(os.path.basename(im.get("file_path", "")))[0]
        boxes = []
        for a in im.get("annotations", []):
            x, y, w, h = a["bbox"]
            boxes.append((x, y, x + w, y + h, int(a["category_id"])))
        if acc:
            out[acc] = boxes
    return out


def load_pred(ans_dir, cid_of_en):
    """{acc: [(x1,y1,x2,y2,cid,conf), ...]} 原图像素。"""
    out = {}
    for fn in sorted(os.listdir(ans_dir)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(ans_dir, fn), encoding="utf-8"))
        P = d.get("Prediction", {})
        size = P.get("ImageSize", {})
        sp = size.get("pixel_spacing_mm", [1.0, 1.0])
        spc = sp[0] if sp and sp[0] else 1.0
        spr = sp[1] if sp and len(sp) > 1 and sp[1] else spc
        acc = d.get("AccessionNumber") or fn[:-5]
        boxes = []
        for det in P.get("Detections", []):
            bb = det.get("BBox_mm")
            cid = cid_of_en.get(det.get("Finding"))
            if not bb or cid is None:
                continue
            boxes.append((bb[0] / spc, bb[1] / spr, bb[2] / spc, bb[3] / spr,
                          cid, det.get("Confidence", 0.0)))
        out[acc] = boxes
    return out


def axis_decomp(ans_dir, gt_ann, cid2en):
    cid_of_en = {e: c for c, e in cid2en.items()}
    gt, pred = load_gt(gt_ann), load_pred(ans_dir, cid_of_en)
    accs = [a for a in gt if a in pred]
    print("[axis] GT %d 张 / answer %d 张 / 交集 %d 张"
          % (len(gt), len(pred), len(accs)))
    if not accs:
        print("[axis] 没有交集 —— 检查 --gt-ann 路径与 answer 目录。")
        return

    per = {}   # cid -> 各统计量列表
    for acc in accs:
        pb = pred[acc]
        for g in gt[acc]:
            cid = g[4]
            cands = [p for p in pb if p[4] == cid]
            if not cands:
                continue
            best = max(cands, key=lambda p: _iou(p[:4], g[:4]))
            gw, gh = g[2] - g[0], g[3] - g[1]
            if gw <= 0 or gh <= 0:
                continue
            r = per.setdefault(cid, dict(iou=[], ix=[], iy=[], hw=[], hh=[],
                                         ox=[], oy=[]))
            r["iou"].append(_iou(best[:4], g[:4]))
            r["ix"].append(_iou_1d(best[0], best[2], g[0], g[2]))
            r["iy"].append(_iou_1d(best[1], best[3], g[1], g[3]))
            r["hw"].append((best[2] - best[0]) / gw)
            r["hh"].append((best[3] - best[1]) / gh)
            r["ox"].append(((best[0] + best[2]) / 2 - (g[0] + g[2]) / 2) / gw)
            r["oy"].append(((best[1] + best[3]) / 2 - (g[1] + g[3]) / 2) / gh)

    if not per:
        print("\n  (answer 里没有任何与 GT 同类的框)")
        return

    hdr = ("%-22s%6s%8s%8s%8s%9s%9s"
           % ("类别", "n配对", "IoU", "iou_x", "iou_y", "宽比w", "高比h"))
    print(); print(hdr); print("-" * len(hdr))
    for cid in sorted(per):
        r = per[cid]
        print("%-22s%6d%8.3f%8.3f%8.3f%9.2f%9.2f"
              % (cid2en.get(cid, str(cid)), len(r["iou"]),
                 _med(r["iou"]), _med(r["ix"]), _med(r["iy"]),
                 _med(r["hw"]), _med(r["hh"])))

    print("\n  归一化中心偏移(相对 GT 自身宽/高, 负=预测偏上/偏左):")
    for cid in sorted(per):
        r = per[cid]
        print("    %-22s  Δcy/h=%+.2f (IQR %.2f)   Δcx/w=%+.2f (IQR %.2f)"
              % (cid2en.get(cid, str(cid)),
                 _med(r["oy"]), _iqr(r["oy"]), _med(r["ox"]), _iqr(r["ox"])))

    # ---- 总判读
    allix = [v for r in per.values() for v in r["ix"]]
    alliy = [v for r in per.values() for v in r["iy"]]
    allhh = [v for r in per.values() for v in r["hh"]]
    mx, my, mh = _med(allix), _med(alliy), _med(allhh)
    print("\n  ===== 判读 =====")
    print("  全部配对: 中位 iou_x=%.3f  iou_y=%.3f  高度比=%.3f  (n=%d)"
          % (mx, my, mh, len(allix)))
    if mx > 0.3 and my < 0.1:
        print("  -> x 轴基本对位, 只有 y 坏 —— 不是整体不行, 是一根轴的问题(有救)。")
    elif my > 0.3 and mx < 0.1:
        print("  -> y 轴基本对位, 只有 x 坏。")
    elif mx < 0.1 and my < 0.1:
        print("  -> 两根轴都不对位 —— 整框崩, 与缩放/翻转无关。")
    else:
        print("  -> 两根轴都有部分对位, 属定位精度问题而非坐标口径问题。")
    print("  高度比 pred_h/gt_h: 中位 %.3f, IQR %.3f" % (mh, _iqr(allhh)))
    if mh < 0.15 and _iqr(allhh) < 0.15:
        print("  -> 预测高度普遍只有 GT 的 ~%.0f%% 且不随 GT 变化 —— "
              "**塌缩**(恒定输出小框), 任何缩放系数都修不了。" % (mh * 100))
        if mx > 0.3:
            print("     问题收窄为「y 方向框高度塌缩」: 先查分辨率(334 token "
                  "下 y 只有 ~560px), 再查 prompt, 最后才是模型能力。")
    elif 0.85 < mh < 1.18:
        print("  -> 高度比接近 1: 尺寸没问题, 偏差来自位置/识别。")
    else:
        print("  -> 高度比偏离 1 且较集中: 疑似固定缩放偏差, 写死换算系数可验证。")


# ---------------------------------------------------------------- mode 4
def side_bias(ans_dir, gt_ann, cid2en):
    """「框偏左/偏上」到底是哪种问题 —— 三个假说, 用数据分开:

      H1 幻觉框落在先验位置: 问一个图里没有的类, grounding 模型(训练时只见过
         阳性样本)照样给一个框, 位置退回先验。 -> 「类不在 GT」的框偏左, 真阳性不偏。
      H2 x 被按长边归一化: 训练时若 x/y 都除以 max(W,H)(pad 成正方形再归一化),
         竖版图上 x 会被整体压缩 W/H 倍、往左挤。 -> 配对框 pred_cx/gt_cx ≈ W/H。
      H3 真阳性也系统性平移。 -> 配对框 Δcx / Δcy 中位数明显偏离 0。
    坐标一律归一化到原图宽高(0~1), 图像左 = 患者右(PA 位)。
    """
    cid_of_en = {e: c for c, e in cid2en.items()}
    gt = load_gt(gt_ann)
    groups = {"GT": [], "pred_present": [], "pred_absent": []}
    absent_by_cls, pairs, absent_raw = {}, [], []
    n_img = 0
    for fn in sorted(os.listdir(ans_dir)):
        if not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(ans_dir, fn), encoding="utf-8"))
        acc = d.get("AccessionNumber") or fn[:-5]
        if acc not in gt:
            continue
        P = d.get("Prediction", {})
        size = P.get("ImageSize", {})
        W, H = size.get("width_px"), size.get("height_px")
        sp = size.get("pixel_spacing_mm") or [1.0, 1.0]
        spc = sp[0] or 1.0
        spr = sp[1] if len(sp) > 1 and sp[1] else spc
        if not W or not H:
            continue
        n_img += 1
        g_boxes = gt[acc]
        present = {g[4] for g in g_boxes}
        for g in g_boxes:
            groups["GT"].append(((g[0] + g[2]) / 2 / W, (g[1] + g[3]) / 2 / H))
        for det in P.get("Detections", []):
            bb, cid = det.get("BBox_mm"), cid_of_en.get(det.get("Finding"))
            if not bb or cid is None:
                continue
            x1, y1, x2, y2 = bb[0] / spc, bb[1] / spr, bb[2] / spc, bb[3] / spr
            c = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)
            pb = (x1, y1, x2, y2)
            if cid in present:
                groups["pred_present"].append(c)
                g = max((g for g in g_boxes if g[4] == cid),
                        key=lambda g: _iou(pb, g[:4]))
                gc = ((g[0] + g[2]) / 2 / W, (g[1] + g[3]) / 2 / H)
                pairs.append((c, gc, W / float(H), pb, g[:4], W, H, cid))
            else:
                groups["pred_absent"].append(c)
                absent_by_cls.setdefault(det.get("Finding"), []).append(c)
                absent_raw.append((pb, W, H))

    print("[side] %s: %d 张有 GT 的图" % (ans_dir, n_img))
    if not n_img:
        print("[side] 没有交集 —— 检查 --gt-ann 与 answer 目录。")
        return
    hdr = "%-26s%6s%10s%10s%14s%14s" % ("组", "n", "中位cx", "中位cy",
                                         "图像左半占比", "图像上半占比")
    print(); print(hdr); print("-" * len(hdr))
    name = {"GT": "GT 框(参照)", "pred_present": "预测·类在GT里",
            "pred_absent": "预测·类不在GT里(幻觉)"}
    left = {}
    for k in ("GT", "pred_present", "pred_absent"):
        cs = groups[k]
        if not cs:
            print("%-26s%6d" % (name[k], 0))
            continue
        lf = sum(1 for c in cs if c[0] < 0.5) / len(cs)
        up = sum(1 for c in cs if c[1] < 0.5) / len(cs)
        left[k] = lf
        print("%-26s%6d%10.3f%10.3f%13.0f%%%13.0f%%"
              % (name[k], len(cs), _med([c[0] for c in cs]),
                 _med([c[1] for c in cs]), 100 * lf, 100 * up))
    n_pred = len(groups["pred_present"]) + len(groups["pred_absent"])
    if n_pred:
        print("  幻觉框占全部预测框: %d/%d (%.0f%%)"
              % (len(groups["pred_absent"]), n_pred,
                 100.0 * len(groups["pred_absent"]) / n_pred))

    if absent_by_cls:
        print("\n  幻觉框按类(n>=2):")
        for en, cs in sorted(absent_by_cls.items(), key=lambda t: -len(t[1])):
            if len(cs) < 2:
                continue
            print("    %-22s n=%-4d 中心=(%.2f,%.2f)  左半占比 %.0f%%"
                  % (en, len(cs), _med([c[0] for c in cs]), _med([c[1] for c in cs]),
                     100.0 * sum(1 for c in cs if c[0] < 0.5) / len(cs)))

    print("\n  ===== 判读 =====")
    # H1
    if "pred_absent" in left and "GT" in left:
        d1 = left["pred_absent"] - left["GT"]
        print("  H1 幻觉框左半占比 - GT 左半占比 = %+.0f 个百分点" % (100 * d1))
        if d1 > 0.15:
            print("     -> **偏左主要来自幻觉框**: 图里没有的类, 模型照样画, 位置退回先验"
                  "(图像左 = 患者右肺)。这不是坐标问题, 是闸门放进了不该放的类。")
    # H2 / H3
    if len(pairs) >= 3:
        rx = [p[0][0] / p[1][0] for p in pairs if p[1][0] > 0.05]
        asp = _med([p[2] for p in pairs])
        dcx = _med([p[0][0] - p[1][0] for p in pairs])
        dcy = _med([p[0][1] - p[1][1] for p in pairs])
        print("  H2 配对框 pred_cx/gt_cx 中位 = %.3f   图像宽高比 W/H 中位 = %.3f   (n=%d)"
              % (_med(rx), asp, len(rx)))
        if asp < 0.95 and abs(_med(rx) - asp) < 0.06 and abs(_med(rx) - 1) > 0.08:
            print("     -> 比值贴近 W/H, 支持「x 按长边归一化」—— 以下方反事实 IoU 为准。")
        elif _med(rx) < 0.92:
            print("     -> 比值明显 <1 但没贴近 W/H: x 有乘性压缩, 原因待反事实 IoU 判。")
        else:
            print("     -> 比值接近 1, 不支持 x 压缩。")
        print("  H3 配对框中心偏移: Δcx 中位 %+.3f  Δcy 中位 %+.3f  (负 = 偏左/偏上)"
              % (dcx, dcy))
        if dcx < -0.05:
            print("     -> 真阳性也系统性偏左。")
        if dcy < -0.05:
            print("     -> 真阳性系统性偏上 %.0f%% 图高。" % (-100 * dcy))
        if dcx >= -0.05 and dcy >= -0.05:
            print("     -> 真阳性没有系统性平移, 位置偏差主要不在坐标链路。")
        _counterfactual(pairs, absent_raw, groups)
    else:
        print("  H2/H3: 同类配对框只有 %d 个, 不够判 —— 多跑几张或放宽 PRESENT_THR。"
              % len(pairs))


def _linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return float("nan"), float("nan")
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return k, my - k * mx


def _longside(box, W, H):
    """把「按 W/H 各自归一化」解码出的框, 改成「x/y 都按 max(W,H) 归一化」解码。
    竖版图上只动 x(乘 H/W), 横版图上只动 y。"""
    L = float(max(W, H))
    sx, sy = L / W, L / H
    x1, y1, x2, y2 = box
    return (min(x1 * sx, W), min(y1 * sy, H), min(x2 * sx, W), min(y2 * sy, H))


def _counterfactual(pairs, absent_raw, groups):
    """反事实: 同一批框换一种解码方式, 看 IoU 涨不涨 —— 比看比值硬。

    关键是对照组要**零参数**: 「按长边解码」不从这批数据里拟合任何东西,
    所以它若显著提高 IoU, 不是过拟合。另给一个用本批数据拟合的「最优 x 缩放」
    当上限参考(会偏乐观)。
    """
    xs = [p[1][0] for p in pairs]; ys_ = [p[0][0] for p in pairs]
    kx, bx = _linfit(xs, ys_)
    ky, by = _linfit([p[1][1] for p in pairs], [p[0][1] for p in pairs])
    print("\n  回归 pred_c = k·gt_c + b (n=%d):" % len(pairs))
    print("    x: k=%.3f b=%+.3f    y: k=%.3f b=%+.3f" % (kx, bx, ky, by))
    print("    (纯乘性压缩 -> k<1 且 b≈0;  纯平移 -> k≈1 且 b<0)")

    def stats(fn):
        ious = [_iou(fn(p), p[4]) for p in pairs]
        return _med(ious), sum(1 for v in ious if v >= 0.3) / len(ious), \
            sum(1 for v in ious if v >= 0.5) / len(ious)

    rx = _med([p[0][0] / p[1][0] for p in pairs if p[1][0] > 0.05])
    sh = _med([p[0][0] - p[1][0] for p in pairs])     # 归一化 x 平移量(负=偏左)

    def _shift(p):
        W = p[5]; d = -sh * W
        return (max(0.0, p[3][0] + d), p[3][1], min(W, p[3][2] + d), p[3][3])

    variants = [
        ("原解码 (现状)", lambda p: p[3]),
        ("x/y 按长边解码 [零参数]", lambda p: _longside(p[3], p[5], p[6])),
        ("x ÷ %.3f [本批拟合缩放]" % rx,
         lambda p: (min(p[3][0] / rx, p[5]), p[3][1], min(p[3][2] / rx, p[5]), p[3][3])),
        ("x 平移 %+.3f [本批拟合平移]" % -sh, _shift),
    ]
    print("\n  反事实 IoU(同类配对框, 同一批模型输出, 只换解码):")
    print("    %-30s%10s%10s%10s" % ("解码方式", "中位IoU", "IoU≥0.3", "IoU≥0.5"))
    res = []
    for name, fn in variants:
        m, a3, a5 = stats(fn)
        res.append((m, a3, a5))
        print("    %-30s%10.3f%9.0f%%%9.0f%%" % (name, m, 100 * a3, 100 * a5))
    base, ls, fsc, fsh = res
    # 判据: 零参数的长边解码必须**不输给**两个用本批数据拟合的变体 ——
    # 单看「比现状好」不够: 纯平移的数据, x 拉伸也能顺带补回一部分 IoU(已用合成数据踩过)。
    best_fit = max(fsc[0], fsh[0])
    if ls[0] - base[0] >= 0.05 and ls[0] >= best_fit - 0.02:
        print("    -> **按长边解码胜出**(零参数, 且不输给本批拟合的缩放/平移):")
        print("       坐标口径是 0-100 × max(W,H), 不是各自除以 W/H。解码 bug, 改 norm100 即可, "
              "不用重跑模型。")
    elif fsh[0] > max(ls[0], fsc[0]) + 0.02:
        print("    -> 拟合平移最好, 长边解码不如它: 是**系统性平移**, 不是归一化口径问题。")
        print("       平移量是本批拟合的, 别直接写进解码 —— 先用另一批图验证是否稳定。")
    elif fsc[0] > max(ls[0], fsh[0]) + 0.02:
        print("    -> 拟合缩放最好但不等于 W/H: 有乘性压缩, 原因不是长边归一化, 待查。")
    elif base[0] >= max(ls[0], best_fit) - 0.02:
        print("    -> 现状解码不输给任何变体: **解码是对的**, 偏差来自模型定位本身。")
    else:
        print("    -> 几种解码差不多: 证据不足, 别动解码。")

    # H1 复核: 把幻觉框也按长边解码, 看偏左是不是还在 —— 若消失, H1 只是 H2 的副产物
    if absent_raw and groups["GT"]:
        lf_gt = sum(1 for c in groups["GT"] if c[0] < 0.5) / len(groups["GT"])
        cs = []
        for pb, W, H in absent_raw:
            b = _longside(pb, W, H)
            cs.append(((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H))
        lf = sum(1 for c in cs if c[0] < 0.5) / len(cs)
        print("\n  H1 复核: 幻觉框按长边解码后, 左半占比 - GT = %+.0f 个百分点 "
              "(原 %+.0f)" % (100 * (lf - lf_gt),
                              100 * (sum(1 for c in groups['pred_absent'] if c[0] < 0.5)
                                     / len(groups['pred_absent']) - lf_gt)))
        if lf - lf_gt < 0.08:
            print("    -> 偏左基本消失: **H1 只是解码压缩的副产物**, 不是独立问题。")
        else:
            print("    -> 解码修正后仍偏左: 幻觉框确实另有先验偏置。")


# ---------------------------------------------------------------- main
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="answer json 或 answer 目录")
    ap.add_argument("--gt-ann", default=None)
    ap.add_argument("--side", action="store_true",
                    help="(需 --gt-ann) 诊断框偏左/偏上: 幻觉框先验 vs 长边归一化 vs 系统平移")
    ap.add_argument("--coord-scale", type=float, default=1.0,
                    help="坐标口径重解释系数(默认 1=不重解释; "
                         "10=把当前框按 0-100 口径重新解释, 零成本验证口径假说)")
    a = ap.parse_args()

    if os.path.isdir(a.path):
        if not a.gt_ann:
            class_geo(a.path, a.coord_scale)   # 无 GT: 框落在哪/多大/散不散
            return
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _shared
        cid2en = {int(c): e for c, e in _shared._cat.CATID_TO_EN.items()}
        if a.side:
            side_bias(a.path, a.gt_ann, cid2en)
        else:
            axis_decomp(a.path, a.gt_ann, cid2en)
    else:
        single(a.path)


if __name__ == "__main__":
    main()
