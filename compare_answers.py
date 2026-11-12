"""
多个 answer 目录在同一份金标准上的指标对拍表。

典型用法: 把 Qwen3.5 零样本结果 和 Faster R-CNN 三组融合结果 并排比(本地服务器路径已是默认值):
  python compare_answers.py --common-only
  # 等价于:
  python compare_answers.py \
    --gt-ann ../track8_detection_sized/pretrain_splits/val.json \
    --image-root /mnt/data/jinghansun/data/match_yibao/reorg \
    --common-only \
    --runs frcnn=../track8_detection_sized/eval_pretrain_val qwen=./answer_qwen

输出:
  1) 总表: S_clin / mAP / APs / APm / APl / S_attr / S_explain
  2) 逐类 AP 并排(含 nGT), 按第一个 run 的 AP 降序 —— 直接看出 Qwen 在哪些类上
     反超/塌陷(经验预期: 心脏增大、纵隔增宽这类"大而规整"的类 VLM 能打,
     结节/钙化这类小目标会接近 0)。
  3) Δ 列(其余 run 相对第一个 run)。

只依赖 numpy。复用 evaluate.score_predictions, 口径与单跑评估完全一致。
"""
import os
import json
import argparse

import _shared  # noqa: F401
import numpy as np
from evaluate import score_predictions


def _f(v, w=6):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a".rjust(w)
    return f"{v:.3f}".rjust(w)


def _d(a, b, w=7):
    if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in (a, b)):
        return "".rjust(w)
    s = f"{a - b:+.3f}"
    return s.rjust(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-ann", default="../track8_detection_sized/pretrain_splits/val.json")
    ap.add_argument("--image-root", default="/mnt/data/jinghansun/data/match_yibao/reorg")
    ap.add_argument("--runs", nargs="+",
                    default=["frcnn=../track8_detection_sized/eval_pretrain_val",
                             "qwen=./answer_qwen"],
                    help="name=dir 形式, 第一个作为基准列")
    ap.add_argument("--common-only", action="store_true",
                    help="只在所有 run 都有预测的图上评估(公平对比, 推荐)")
    ap.add_argument("--no-mm-convert", action="store_true")
    ap.add_argument("--json-out", default=None, help="同时把结果写一份 json")
    args = ap.parse_args()

    runs = []
    for r in args.runs:
        if "=" not in r:
            raise SystemExit(f"--runs 需要 name=dir 形式, 收到: {r}")
        name, d = r.split("=", 1)
        if not os.path.isdir(d):
            raise SystemExit(f"目录不存在: {d}")
        runs.append((name, d))

    with open(args.gt_ann, "r", encoding="utf-8") as f:
        images = json.load(f).get("images", [])

    if args.common_only:
        from evaluate import acc_from_image
        sets = []
        for _, d in runs:
            sets.append({os.path.splitext(fn)[0] for fn in os.listdir(d) if fn.endswith(".json")})
        common = set.intersection(*sets) if sets else set()
        images = [im for im in images if acc_from_image(im) in common]
        print(f"[common-only] 交集图像数: {len(images)}")

    results = []
    for name, d in runs:
        print(f"\n########## scoring run: {name}  ({d}) ##########")
        results.append((name, score_predictions(d, images, args.image_root,
                                                args.no_mm_convert, verbose=False)))

    # ---- 总表 ----
    print("\n" + "=" * (26 + 10 * len(results)))
    print("总表 (官方 S_clin 口径)")
    print("=" * (26 + 10 * len(results)))
    hdr = f"{'指标':<22s}" + "".join(f"{n:>10s}" for n, _ in results)
    print(hdr)
    for key, label in (("S_clin", "S_clin (总)"), ("mAP", "mAP (17类, 60%)"),
                       ("APs", "APs 小<30mm²"), ("APm", "APm 中30-300"),
                       ("APl", "APl 大>=300"), ("S_attr", "S_attr (25%)"),
                       ("S_explain", "S_explain IoU (15%)"),
                       ("mass_relerr_median", "肿块相对误差中位"),
                       ("eff_kappa", "积液加权Kappa"), ("n_eval", "评估图数")):
        cells = ""
        for _, r in results:
            v = r.get(key)
            cells += (f"{v:>10d}" if key == "n_eval" else _f(v, 10))
        print(f"{label:<22s}{cells}")

    # ---- 逐类 AP ----
    base_name, base = results[0]
    print("\n" + "=" * (40 + 18 * (len(results) - 1)))
    print(f"逐类 AP 对拍 (基准 = {base_name}; Δ 为其余 run 减基准)")
    print("=" * (40 + 18 * (len(results) - 1)))
    hdr = f"{'Finding':<22s}{'nGT':>6s}" + f"{base_name:>9s}"
    for n, _ in results[1:]:
        hdr += f"{n:>9s}{'Δ':>8s}"
    print(hdr)
    rows = sorted(base["per_class"],
                  key=lambda r: -(r["ap"] if r["ap"] == r["ap"] else -1.0))
    by_name = [{r["en"]: r for r in res["per_class"]} for _, res in results]
    for row in rows:
        en = row["en"]
        line = f"{en:<22s}{row['ngt']:>6d}{_f(row['ap'], 9)}"
        for k in range(1, len(results)):
            o = by_name[k].get(en, {}).get("ap")
            line += _f(o, 9) + _d(o, row["ap"], 8)
        print(line)
    print("=" * (40 + 18 * (len(results) - 1)))
    print("注: AP=n/a 表示该类在当前评估子集里无 GT, 不计入 mAP。")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({n: r for n, r in results}, f, ensure_ascii=False,
                      indent=2, default=lambda o: None if o != o else o)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()
