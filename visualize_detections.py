#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
离线批量可视化 Qwen 检测框 —— 不需要 GPU / 不需要 DICOM / 不加载模型。

只吃两样东西(都是 qwen_infer.py 跑完就落在工作目录的现成产物):
  --png-dir     png_qwen/   : 模型实际看到的 PNG(dicom_to_png_file 导出, 已限长边)
  --answer-dir  answer_qwen/: 每张图一个 {acc}.json, 内含 Prediction.Detections[].BBox_mm

坐标换算: BBox_mm --(/pixel_spacing)--> 原图像素 --(×png/原图比例)--> PNG 像素,
与 inference_sized / probe_one 同一套口径。--with-gt 时再把 val.json 的 GT 框叠上
(左 = 预测, 右 = 金标准; 不给 --gt-ann 则只画预测, 单栏)。

典型用法(服务器上, 数据就在本地目录):
  python -u visualize_detections.py --n 12 --with-gt \
      --gt-ann ../track8_detection_sized/pretrain_splits/val.json
  # 只看某一张:
  python -u visualize_detections.py --acc ZHDR202107161121 --with-gt --gt-ann ...
  # 只画高置信框、每图最多 8 个:
  python -u visualize_detections.py --min-conf 0.3 --max-box 8

产物: --out-dir(默认 ./vis_det)下每张图一个 {acc}_overlay.png, 外加一张
       contact_sheet.png 缩略拼图, 方便一眼扫一批。
"""
import os
import sys
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image
    _HAS_PLOT = True
except Exception as e:  # pragma: no cover
    _HAS_PLOT = False
    _IMPORT_ERR = e

# 17 类中英 + 固定配色(与 PPT 一致的冷暖区分: 局灶暖色 / 弥漫冷色)
try:
    from _shared import _cat
    CATID_TO_EN = {i: _cat.CATID_TO_EN[i] for i in range(len(_cat.CATID_TO_EN))}
    EN_TO_CATID = dict(_cat.EN_TO_CATID)
except Exception:
    CATID_TO_EN = {}
    EN_TO_CATID = {}

_PALETTE = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4", "#42D4F4",
    "#F032E6", "#BFEF45", "#FABED4", "#469990", "#DCBEFF", "#9A6324",
    "#800000", "#AAFFC3", "#808000", "#FFD8B1", "#000075",
]


def _color_for_cid(cid):
    """每个类别一个固定颜色 —— pred 和 GT 的同类框用同色, 便于肉眼对位。"""
    if cid is None:
        return "#CCCCCC"
    return _PALETTE[cid % len(_PALETTE)]


def _acc_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def _find_png(png_dir, acc):
    for cand in (f"{acc}.png", f"{acc}.PNG"):
        p = os.path.join(png_dir, cand)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(png_dir, f"{acc}*.png"))
    return hits[0] if hits else None


def _load_gt(gt_ann):
    """返回 {acc: [(x1,y1,x2,y2,cid), ...]}  (原图像素)。"""
    if not gt_ann or not os.path.exists(gt_ann):
        return {}
    with open(gt_ann, "r", encoding="utf-8") as f:
        data = json.load(f)
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


def _pred_boxes_px(ans):
    """从 answer json 取 (x1,y1,x2,y2, label, conf)  原图像素。"""
    pr = ans.get("Prediction", {})
    size = pr.get("ImageSize", {})
    W0 = size.get("width_px"); H0 = size.get("height_px")
    sp = size.get("pixel_spacing_mm", [1.0, 1.0])
    sp_col = sp[0] if sp and sp[0] else 1.0
    sp_row = sp[1] if sp and len(sp) > 1 and sp[1] else sp_col
    out = []
    for d in pr.get("Detections", []):
        bb = d.get("BBox_mm")
        if not bb or len(bb) != 4:
            continue
        x1 = bb[0] / sp_col; y1 = bb[1] / sp_row
        x2 = bb[2] / sp_col; y2 = bb[3] / sp_row
        out.append((x1, y1, x2, y2, d.get("Finding", "?"),
                    d.get("Confidence")))
    return out, W0, H0


def _draw_boxes(ax, boxes, W0, H0, pw, ph, mode, emphasize_cid=None):
    """boxes: pred=(x1,y1,x2,y2,label,conf) 或 gt=(x1,y1,x2,y2,cid)。按比例缩到 PNG。
    颜色由类别决定(同类同色); 线型区分: 预测=实线, GT=虚线。
    emphasize_cid: 逐类模式下把该类的框加粗(其余类细线), 突出当前关注类。"""
    sx = pw / float(W0) if W0 else 1.0
    sy = ph / float(H0) if H0 else 1.0
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        if mode == "pred":
            label, conf = b[4], b[5]
            cid = EN_TO_CATID.get(label, EN_TO_CATID.get(str(label).strip(), None))
            col = _color_for_cid(cid) if cid is not None else _PALETTE[i % len(_PALETTE)]
            tag = f"{label}" + (f" {conf:.2f}" if isinstance(conf, (int, float)) else "")
            ls = "solid"
        else:
            cid = b[4]
            label = CATID_TO_EN.get(cid, str(cid))
            col = _color_for_cid(cid)
            tag = f"GT:{label}"
            ls = (0, (4, 2))   # 虚线
        if emphasize_cid is None:
            lw, alpha = 2.0, 1.0
        elif cid == emphasize_cid:
            lw, alpha = 3.2, 1.0          # 当前关注类: 粗、实
        else:
            lw, alpha = 1.3, 0.75         # 其余类: 细、略透明, 当上下文
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                     fill=False, edgecolor=col, linewidth=lw,
                     linestyle=ls, alpha=alpha))
        ax.text(x1, max(0, y1 - 4), tag, fontsize=8, color="white",
                bbox=dict(boxstyle="square,pad=0.1", fc=col, ec="none",
                          alpha=0.88 if (emphasize_cid is None or cid == emphasize_cid) else 0.6))


def render_one(acc, png_path, ans, gt_boxes, out_dir, with_gt,
               min_conf, max_box, tag="Prediction"):
    im = Image.open(png_path).convert("RGB")
    pw, ph = im.size
    preds, W0, H0 = _pred_boxes_px(ans)
    if not W0:
        W0, H0 = pw, ph
    if min_conf > 0:
        preds = [p for p in preds
                 if (p[5] is None) or (p[5] >= min_conf)]
    preds = sorted(preds, key=lambda p: -(p[5] or 0))[:max_box] if max_box else preds

    panels = [(f"{tag} prediction", preds, "pred")]
    if with_gt:
        panels.append(("Ground Truth", gt_boxes or [], "gt"))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(6.4 * len(panels), 7.2), dpi=120)
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, boxes, mode) in zip(axes, panels):
        ax.imshow(im)
        _draw_boxes(ax, boxes, W0, H0, pw, ph, mode)
        n = len(boxes)
        ax.set_title(f"{title}  ({n} boxes)", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{acc}", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, f"{acc}_overlay.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, len(preds)


def contact_sheet(items, out_dir, cols=4, fname="contact_sheet.png"):
    """items: [(acc, overlay_path)]  拼一张缩略总览。"""
    if not items:
        return None
    n = len(items)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.6), dpi=100)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax in axes:
        ax.axis("off")
    for ax, (acc, p) in zip(axes, items):
        try:
            ax.imshow(Image.open(p))
            ax.set_title(acc, fontsize=8)
        except Exception:
            pass
    out = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ==========================================================================
# 逐类可视化: 每个类别挑几张"GT 含该类"的图, 只画该类的框(pred 暖色 / GT 青)
# ==========================================================================
def _pred_boxes_with_cid(ans):
    """复用 _pred_boxes_px, 再把 label 映射成 cid。返回 (list[(x1,y1,x2,y2,label,conf,cid)], W0, H0)。"""
    preds, W0, H0 = _pred_boxes_px(ans)
    out = []
    for x1, y1, x2, y2, label, conf in preds:
        cid = EN_TO_CATID.get(label, EN_TO_CATID.get(str(label).strip(), None))
        out.append((x1, y1, x2, y2, label, conf, cid))
    return out, W0, H0


def do_per_class(args):
    tag = getattr(args, "tag", "Prediction")
    gt = _load_gt(args.gt_ann)
    if not gt:
        print(f"[ERROR] 逐类模式必须给 --gt-ann(用 GT 来确定每张图含哪些类): "
              f"{args.gt_ann}")
        sys.exit(1)
    if not EN_TO_CATID:
        print("[ERROR] 没加载到类别表(_shared._cat), 无法把预测标签映射到 cid")
        sys.exit(1)

    ans_files = sorted(glob.glob(os.path.join(args.answer_dir, "*.json")))
    if not ans_files:
        print(f"[ERROR] {args.answer_dir} 下没有 answer json")
        sys.exit(1)
    have_ans = {_acc_of(f): f for f in ans_files}

    # 建 cid -> [acc] 候选(GT 含该类, 且有 answer + png)
    import random
    rng = random.Random(args.seed)
    by_cls = {c: [] for c in range(len(CATID_TO_EN))}
    for acc, boxes in gt.items():
        if acc not in have_ans:
            continue
        if not _find_png(args.png_dir, acc):
            continue
        for c in {b[4] for b in boxes}:
            if c in by_cls:
                by_cls[c].append(acc)

    os.makedirs(args.out_dir, exist_ok=True)
    reps, empty = [], []
    summary = []
    for cid in range(len(CATID_TO_EN)):
        en = CATID_TO_EN.get(cid, str(cid))
        cands = sorted(set(by_cls.get(cid, [])))
        if not cands:
            empty.append(en)
            summary.append((cid, en, 0))
            continue
        rng.shuffle(cands)
        pick = cands[:args.per_class_n]
        cls_dir = os.path.join(args.out_dir, f"{cid:02d}_{en}")
        os.makedirs(cls_dir, exist_ok=True)
        for acc in pick:
            png = _find_png(args.png_dir, acc)
            with open(have_ans[acc], "r", encoding="utf-8") as f:
                ans = json.load(f)
            preds_cid, W0, H0 = _pred_boxes_with_cid(ans)
            if not W0:
                im0 = Image.open(png); W0, H0 = im0.size
            # 默认画全图所有框(target 类加粗突出); --only-target 才只画该类
            if args.only_target:
                p_draw = [(p[0], p[1], p[2], p[3], p[4], p[5])
                          for p in preds_cid if p[6] == cid]
                g_draw = [b for b in gt.get(acc, []) if b[4] == cid]
                emph = None
            else:
                p_draw = [(p[0], p[1], p[2], p[3], p[4], p[5]) for p in preds_cid]
                g_draw = gt.get(acc, [])
                emph = cid

            im = Image.open(png).convert("RGB")
            pw, ph = im.size
            n_pt = sum(1 for p in preds_cid if p[6] == cid)
            n_gt = sum(1 for b in gt.get(acc, []) if b[4] == cid)
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=120)
            for ax, (ttl, boxes, mode) in zip(
                    axes, [(f"{tag} pred (focus: {en})", p_draw, "pred"),
                           (f"GT (focus: {en})", g_draw, "gt")]):
                ax.imshow(im)
                _draw_boxes(ax, boxes, W0, H0, pw, ph, mode, emphasize_cid=emph)
                ax.set_title(f"{ttl}  [{len(boxes)} boxes]", fontsize=12)
                ax.axis("off")
            fig.suptitle(f"[{cid:02d}] {en}   {acc}   "
                         f"(pred_{en}={n_pt}, gt_{en}={n_gt})", fontsize=13, y=0.99)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            outp = os.path.join(cls_dir, f"{acc}_overlay.png")
            fig.savefig(outp, bbox_inches="tight")
            plt.close(fig)
            print(f"[ok] [{cid:02d}] {en:<20s} {acc}  pred_{en}={n_pt} gt_{en}={n_gt} "
                  f"(全图 pred={len(p_draw)} gt={len(g_draw)})")
            reps.append((f"{en}:{acc}", outp)) if acc == pick[0] else None
        summary.append((cid, en, len(pick)))

    sheet = None
    if reps and not args.no_sheet:
        # 每类第一张拼一张总览(17 格)
        sheet = contact_sheet(reps, args.out_dir, cols=4,
                              fname="per_class_overview.png")

    print("\n" + "=" * 56)
    print("逐类采样结果(GT 含该类的图里各取几张):")
    for cid, en, k in summary:
        flag = "  <-- val 上无 GT, 跳过" if k == 0 else ""
        print(f"  [{cid:02d}] {en:<22s} {k} 张{flag}")
    if empty:
        print(f"\n无样本类({len(empty)}): {', '.join(empty)}  (这些类在 val 上没有 GT 框)")
    print(f"\n输出: {os.path.abspath(args.out_dir)}  (每类一个子目录)")
    if sheet:
        print(f"每类一张的总览: {sheet}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(
        description="离线批量可视化 Qwen 检测框(不需 GPU/DICOM)")
    ap.add_argument("--answer-dir", default="./answer_qwen")
    ap.add_argument("--png-dir", default="./png_qwen")
    ap.add_argument("--out-dir", default="./vis_det")
    ap.add_argument("--gt-ann", default=None, help="给了并开 --with-gt 则叠金标准框")
    ap.add_argument("--with-gt", action="store_true")
    ap.add_argument("--acc", default=None, help="只画这个 AccessionNumber")
    ap.add_argument("--n", type=int, default=12, help="最多画几张(0=全部)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="过滤低置信预测框")
    ap.add_argument("--max-box", type=int, default=0, help=">0 每图最多画几个预测框(按conf降序)")
    ap.add_argument("--no-sheet", action="store_true", help="不生成缩略拼图")
    ap.add_argument("--per-class", action="store_true",
                    help="逐类模式: 每个类别各挑几张(GT 含该类), 只画该类的框。需 --gt-ann")
    ap.add_argument("--per-class-n", type=int, default=3, help="逐类模式下每类画几张")
    ap.add_argument("--all-boxes", action="store_true",
                    help="(已默认)逐类模式画全图所有框; 保留此开关仅为兼容, 无额外效果")
    ap.add_argument("--only-target", action="store_true",
                    help="逐类模式下只画当前类的框(默认画全图所有框, target 类加粗突出)")
    ap.add_argument("--seed", type=int, default=42, help="逐类采样的随机种子(无偏且可复现)")
    ap.add_argument("--tag", default="Prediction",
                    help="图例里写谁的名字(chexone-vis 会传 CheXOne)")
    args = ap.parse_args()

    if not _HAS_PLOT:
        print(f"[ERROR] 需要 matplotlib + pillow: {_IMPORT_ERR}")
        sys.exit(2)
    if not os.path.isdir(args.answer_dir):
        print(f"[ERROR] answer 目录不存在: {args.answer_dir}  "
              f"(先在服务器上跑 run_qwen_eval.sh full / smoke)")
        sys.exit(1)
    if not os.path.isdir(args.png_dir):
        print(f"[ERROR] png 目录不存在: {args.png_dir}")
        sys.exit(1)

    if args.per_class:
        do_per_class(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    gt = _load_gt(args.gt_ann) if args.with_gt else {}
    if args.with_gt and not gt:
        print("[warn] --with-gt 但没读到 GT(未给 --gt-ann 或路径不存在), 只画预测")

    ans_files = sorted(glob.glob(os.path.join(args.answer_dir, "*.json")))
    if args.acc:
        ans_files = [f for f in ans_files if _acc_of(f) == args.acc]
    if not ans_files:
        print(f"[ERROR] {args.answer_dir} 下没有匹配的 answer json")
        sys.exit(1)
    if args.n and args.n > 0:
        ans_files = ans_files[:args.n]

    done, miss = [], 0
    for af in ans_files:
        acc = _acc_of(af)
        png = _find_png(args.png_dir, acc)
        if not png:
            miss += 1
            print(f"[skip] 找不到 png: {acc}")
            continue
        with open(af, "r", encoding="utf-8") as f:
            ans = json.load(f)
        out, npred = render_one(acc, png, ans, gt.get(acc), args.out_dir,
                                args.with_gt, args.min_conf, args.max_box,
                                tag=args.tag)
        print(f"[ok] {acc}  预测{npred}框 -> {out}")
        done.append((acc, out))

    sheet = None
    if done and not args.no_sheet:
        sheet = contact_sheet(done, args.out_dir)

    print("\n" + "=" * 52)
    print(f"画好 {len(done)} 张, 跳过 {miss} 张(缺 png)")
    print(f"输出目录: {os.path.abspath(args.out_dir)}")
    if sheet:
        print(f"缩略拼图: {sheet}")
    print("=" * 52)


if __name__ == "__main__":
    main()
