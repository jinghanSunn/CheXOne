"""逐类 AP + IoU 总表 —— 与官方 evaluate.py 同一套 GT/换算/NMS/阈值, 只是多报 IoU。

为什么不直接看 evaluate.py:
  官方只给 mAP 和 S_explain(= **匹配成功框**的平均 IoU)。后者只在 TP 上取平均,
  模型越不准、TP 越少, 它反而可能越高 —— 不能用来回答"框画得准不准"。
  这里补一个 **GT 视角**的 IoU: 每个 GT 框取同类预测里 IoU 最大的那个,
  没有同类预测记 0。这样漏检会被算进去, 数值不会被幸存者偏差抬高。

列含义:
  nGT / nPred      GT 框数 / NMS 后预测框数
  AP               官方 11 点 AP(直接调 evaluate.score_predictions, 不另算)
  TP               官方贪心匹配下 IoU >= 该类官方阈值的预测数
  R@thr            TP / nGT(官方阈值: 局灶类按面积 0.30/0.40/0.50, 弥漫类 0.25)
  GT-mIoU          GT 视角平均最佳 IoU(漏检记 0)   <- 主看这个
  GT-medIoU        同上取中位数
  R@0.3 / R@0.5    GT 视角最佳 IoU >= 0.3 / 0.5 的比例(与类别阈值无关, 跨类可比)
  TP-mIoU          匹配成功框的平均 IoU(= 官方 S_explain 的逐类版)

用法:
  python iou_report.py --pred-dir answer_chexone \\
      --gt-ann ../track8_detection_sized/pretrain_splits/val.json \\
      --image-root /mnt/data/jinghansun/data/match_yibao/reorg \\
      --out-prefix logs/chexone/chexone_iou
"""
import os
import sys
import csv
import json
import argparse

import numpy as np

import _shared  # noqa: F401  (把 track8_detection_sized 挂进 sys.path)
from evaluate import (score_predictions, acc_from_image, get_spacing, iou_xyxy,
                      box_area, size_bucket, nms_per_class)
from categories import (NUM_CLASSES, CATID_TO_EN, CATID_TO_CN, EN_TO_CATID,
                        iou_threshold_for, NMS_IOU)


def collect(pred_dir, images, image_root, no_mm_convert=False):
    """-> per_cls[cid] = dict(gt_best=[...], tp_iou=[...], n_pred, n_tp),
          per_size[s] = 同结构, n_img"""
    def blank():
        return {"gt_best": [], "tp_iou": [], "n_pred": 0, "n_tp": 0, "n_gt": 0}
    per_cls = {c: blank() for c in range(NUM_CLASSES)}
    per_size = {s: blank() for s in ("s", "m", "l")}
    n_img = 0
    for im in images:
        acc = acc_from_image(im)
        pp = os.path.join(pred_dir, f"{acc}.json")
        if not os.path.exists(pp):
            continue
        n_img += 1
        with open(pp, "r", encoding="utf-8") as f:
            pred = json.load(f)
        if no_mm_convert:
            sx, sy = 1.0, 1.0
        else:
            sx, sy = get_spacing(image_root, im.get("file_path") or im.get("file_name") or "")

        gts = []
        for a in im.get("annotations", []):
            gx, gy, gw, gh = a["bbox"]
            gb = [gx * sx, gy * sy, (gx + gw) * sx, (gy + gh) * sy]
            cid = int(a["category_id"])
            gts.append({"box": gb, "cid": cid, "en": CATID_TO_EN.get(cid),
                        "area": box_area(gb), "matched": False})

        raw = []
        for d in pred.get("Prediction", {}).get("Detections", []):
            cid = EN_TO_CATID.get(d.get("Finding"), -1)
            if cid < 0 or not d.get("BBox_mm"):
                continue
            raw.append({"box": d["BBox_mm"], "en": d["Finding"], "cid": cid,
                        "conf": float(d.get("Confidence", 0.0))})
        preds = nms_per_class(raw, NMS_IOU)
        preds.sort(key=lambda p: -p["conf"])
        for p in preds:
            per_cls[p["cid"]]["n_pred"] += 1
            # 与官方 APs/APm/APl 一致: 预测按**自身**面积分档
            per_size[size_bucket(box_area(p["box"]))]["n_pred"] += 1

        # (1) GT 视角: 每个 GT 的同类最佳 IoU(不消耗匹配, 漏检记 0)
        for g in gts:
            same = [iou_xyxy(p["box"], g["box"]) for p in preds if p["cid"] == g["cid"]]
            best = max(same) if same else 0.0
            sb = size_bucket(g["area"])
            for bucket in (per_cls[g["cid"]], per_size[sb]):
                bucket["gt_best"].append(best)
                bucket["n_gt"] += 1

        # (2) 官方贪心匹配(与 evaluate.score_predictions 逐行一致) -> TP 与 TP-IoU
        for p in preds:
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gts):
                if g["matched"] or g["cid"] != p["cid"]:
                    continue
                v = iou_xyxy(p["box"], g["box"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_threshold_for(p["en"], gts[best_j]["area"]):
                gts[best_j]["matched"] = True
                sb = size_bucket(gts[best_j]["area"])
                for bucket in (per_cls[p["cid"]], per_size[sb]):
                    bucket["tp_iou"].append(best_iou)
                    bucket["n_tp"] += 1
    return per_cls, per_size, n_img


def _row(name, cn, b, ap):
    g = np.array(b["gt_best"], dtype=float)
    n = b["n_gt"]
    return {
        "class": name, "cn": cn, "nGT": n, "nPred": b["n_pred"],
        "AP": ap, "TP": b["n_tp"],
        "R@thr": (b["n_tp"] / n) if n else float("nan"),
        "GT-mIoU": float(g.mean()) if n else float("nan"),
        "GT-medIoU": float(np.median(g)) if n else float("nan"),
        "R@0.3": float((g >= 0.3).mean()) if n else float("nan"),
        "R@0.5": float((g >= 0.5).mean()) if n else float("nan"),
        "TP-mIoU": float(np.mean(b["tp_iou"])) if b["tp_iou"] else float("nan"),
    }


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{100 * v:.1f}%" if pct else (f"{v:.3f}" if isinstance(v, float) else str(v))


COLS = [("class", 22, False), ("nGT", 6, False), ("nPred", 7, False), ("AP", 7, False),
        ("TP", 5, False), ("R@thr", 8, True), ("GT-mIoU", 9, False),
        ("GT-medIoU", 10, False), ("R@0.3", 8, True), ("R@0.5", 8, True),
        ("TP-mIoU", 9, False)]


def _print_table(rows, title):
    print("\n" + title)
    hdr = "".join(f"{c:<{w}s}" if i == 0 else f"{c:>{w}s}" for i, (c, w, _) in enumerate(COLS))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = ""
        for i, (c, w, pct) in enumerate(COLS):
            s = r[c] if c == "class" else _fmt(r[c], pct)
            line += f"{s:<{w}s}" if i == 0 else f"{s:>{w}s}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description="逐类 AP + IoU 总表(官方同口径)")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-ann", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--no-mm-convert", action="store_true")
    ap.add_argument("--only-done", action="store_true", default=True,
                    help="只评有 answer 的图(默认开; 关掉则缺 answer 的图按全漏检计)")
    ap.add_argument("--out-prefix", default=None,
                    help="写 {prefix}.csv / {prefix}.json; 不给则只打印")
    a = ap.parse_args()

    with open(a.gt_ann, "r", encoding="utf-8") as f:
        images = json.load(f).get("images", [])
    done = {os.path.splitext(fn)[0] for fn in os.listdir(a.pred_dir) if fn.endswith(".json")}
    if a.only_done:
        images = [im for im in images if acc_from_image(im) in done]
    if not images:
        print("[iou] answer 与 GT 无交集 —— 检查 --pred-dir / --gt-ann")
        return 1

    off = score_predictions(a.pred_dir, images, a.image_root,
                            no_mm_convert=a.no_mm_convert, verbose=False)
    ap_by_en = {r["en"]: r["ap"] for r in off["per_class"]}
    per_cls, per_size, n_img = collect(a.pred_dir, images, a.image_root, a.no_mm_convert)

    rows = []
    for c in range(NUM_CLASSES):
        en = CATID_TO_EN.get(c, str(c))
        if per_cls[c]["n_gt"] == 0 and per_cls[c]["n_pred"] == 0:
            continue
        rows.append(_row(en, CATID_TO_CN.get(c, ""), per_cls[c], ap_by_en.get(en)))
    rows.sort(key=lambda r: -(r["GT-mIoU"] if not np.isnan(r["GT-mIoU"]) else -1))

    tot = {"gt_best": [], "tp_iou": [], "n_pred": 0, "n_tp": 0, "n_gt": 0}
    for b in per_cls.values():
        tot["gt_best"] += b["gt_best"]; tot["tp_iou"] += b["tp_iou"]
        for k in ("n_pred", "n_tp", "n_gt"):
            tot[k] += b[k]
    overall = _row("ALL (按框合并)", "", tot, off["mAP"])
    valid = [r for r in rows if r["nGT"] > 0]
    macro = {"class": "MACRO (按类平均)", "cn": "", "nGT": sum(r["nGT"] for r in valid),
             "nPred": sum(r["nPred"] for r in valid), "AP": off["mAP"],
             "TP": sum(r["TP"] for r in valid)}
    for k in ("R@thr", "GT-mIoU", "GT-medIoU", "R@0.3", "R@0.5", "TP-mIoU"):
        vs = [r[k] for r in valid if not np.isnan(r[k])]
        macro[k] = float(np.mean(vs)) if vs else float("nan")

    size_ap = {"s": off["APs"], "m": off["APm"], "l": off["APl"]}
    size_rows = [_row({"s": "small <30mm²", "m": "mid 30-300", "l": "large ≥300"}[s], "",
                      per_size[s], size_ap[s]) for s in ("s", "m", "l")]

    print("=" * 100)
    print(f"[iou] {a.pred_dir}: {n_img} 张图 / GT {tot['n_gt']} 框 / 预测 {tot['n_pred']} 框(NMS 后)")
    print(f"      官方: mAP={off['mAP']:.4f}  APs={_fmt(off['APs'])}  APm={_fmt(off['APm'])}  "
          f"APl={_fmt(off['APl'])}  S_explain(TP-mIoU)={off['S_explain']:.4f}  "
          f"S_clin={off['S_clin']:.4f}")
    print("=" * 100)
    _print_table(rows, "[逐类]  按 GT-mIoU 降序   (AP 为官方值; mAP 只对 nGT>0 的类平均)")
    _print_table([macro, overall], "[汇总]  AP 列 = 官方 mAP")
    _print_table(size_rows, "[按 GT 面积分档]  AP 列 = 官方 APs/APm/APl")
    print("\n读法: GT-mIoU 是「框准不准」的主指标(漏检记 0); TP-mIoU 只在命中框上平均,"
          " 命中越少越容易虚高, 别单看它。")

    if a.out_prefix:
        os.makedirs(os.path.dirname(a.out_prefix) or ".", exist_ok=True)
        allrows = rows + [macro, overall] + size_rows
        with open(a.out_prefix + ".csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[c for c, _, _ in COLS] + ["cn"])
            w.writeheader()
            for r in allrows:
                w.writerow({k: r.get(k) for k in w.fieldnames})
        with open(a.out_prefix + ".json", "w", encoding="utf-8") as f:
            json.dump({"n_img": n_img, "official": {k: off[k] for k in
                       ("mAP", "APs", "APm", "APl", "S_attr", "S_explain", "S_clin")},
                       "per_class": rows, "macro": macro, "overall": overall,
                       "per_size": size_rows}, f, ensure_ascii=False, indent=2,
                      default=lambda v: None)
        print(f"\n[iou] 已写 {a.out_prefix}.csv / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
