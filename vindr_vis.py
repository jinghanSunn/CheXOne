"""Visualize CheXOne on VinDr-CXR from the answer cache. No GPU, no DICOM, no model.

Reads what a finished run already left on disk:
  png_vindr/{image_id}.png + .info.json     input image + original size
  cache_vindr/{image_id}__{class}__{fp}.txt raw model answer (reasoning included)
  physio/annotations_test.csv               GT boxes

Outputs under <out>/vis/:
  scatter.png            pred center vs GT center, x and y, with y=x and the fitted line.
                         This is the picture behind the regression k: k near 1 means the
                         model follows the lesion, k near 0 means it ignores it.
  NNN_<class>_<id>_iouX.XX.png
                         one figure per (image, class) query. Left: image, GT dashed green,
                         prediction solid red. Right: GT and prediction in the model's own
                         0-100 coordinates, then the raw answer text.

--reasoning must match the run you want to look at: the cache key contains the prompt, so
the wrong flag finds no cache and draws nothing.

Usage (see run_vindr_eval.sh vis):
  python vindr_vis.py --data-root /mnt/data/jinghansun/data/Vin-DR --out vindr_result_reason --reasoning
"""
import os
import sys
import json
import argparse
import textwrap
import random as _random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import vindr_chexone as V

PICKS = ("per-class", "random", "best", "miss", "nobox")


def load_rows(args, pairs):
    """-> rows, n_missing_cache, n_missing_png. One row per (image, class) with a cached answer."""
    by_img = defaultdict(list)
    for (image_id, cls), gts in pairs.items():
        by_img[image_id].append((cls, gts))
    variant = V.cache_variant(args)
    rows, miss_cache, miss_png = [], 0, 0
    for image_id in sorted(by_img):
        png = os.path.join(args.png_dir, f"{image_id}.png")
        meta = os.path.join(args.png_dir, f"{image_id}.info.json")
        if not (os.path.exists(png) and os.path.exists(meta)):
            miss_png += 1
            continue
        info = None
        for cls, gts in sorted(by_img[image_id]):
            if args.vis_class and cls != args.vis_class:
                continue
            cp = V.cache_path(args.cache_dir, image_id, cls,
                              V.build_prompt(cls, args.prompt_name), variant)
            if not os.path.exists(cp):
                miss_cache += 1
                continue
            if info is None:
                with open(meta, "r", encoding="utf-8") as f:
                    info = json.load(f)
            with open(cp, "r", encoding="utf-8") as f:
                text = f.read()
            rec = V.score_pair(image_id, cls, text, gts, info, {})
            W0, H0 = info["W0"], info["H0"]
            preds, oob = [], []
            for _, b in V.extract_grounding(text):
                px = V.to_pixel(b, V.COORD_MODE, info["W_in"], info["H_in"], W0, H0)
                (oob if px is None else preds).append(b if px is None else px)
            rec.update(text=text, png=png, gts=gts, preds=preds, oob=oob, W0=W0, H0=H0)
            rows.append(rec)
    return rows, miss_cache, miss_png


def pick(rows, mode, n, seed):
    rnd = _random.Random(seed)

    def sample(pool, k):
        pool = list(pool)
        rnd.shuffle(pool)
        return pool[:k]

    if mode == "random":
        return sample(rows, n)
    if mode == "best":
        return sorted(rows, key=lambda r: -r["iou_first"])[:n]
    if mode == "miss":      # a valid box was drawn, but it does not touch any GT
        return sample([r for r in rows if r["preds"] and r["iou_first"] == 0], n)
    if mode == "nobox":     # no usable box: either no answer box or out of range
        return sample([r for r in rows if not r["preds"]], n)
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    out = []
    for cls in sorted(by_cls):
        out += sample(by_cls[cls], n)
    return out


def _wrap(text, width=62, head=800, tail=500):
    text = text.replace("$", r"\$")
    if len(text) > head + tail + 40:
        text = text[:head] + "\n[... truncated ...]\n" + text[-tail:]
    return "\n".join(textwrap.fill(p, width) if p.strip() else "" for p in text.split("\n"))


def draw(r, k, out_dir, all_boxes=False):
    """Draws only the first predicted box by default: that is the one every metric scores.
    all_boxes=True draws the rest too (dashed), to inspect repetition loops."""
    im = Image.open(r["png"]).convert("RGB")
    pw, ph = im.size
    W0, H0 = r["W0"], r["H0"]
    sx, sy = pw / float(W0), ph / float(H0)
    fig, (ax, at) = plt.subplots(1, 2, figsize=(12.5, 6.8),
                                 gridspec_kw={"width_ratios": [1, 1.05]})
    ax.imshow(im)
    # GT thick underneath, prediction thin on top: when the two nearly coincide (the case
    # worth checking) both stay visible instead of one hiding the other.
    for g in r["gts"]:
        ax.add_patch(mpatches.Rectangle((g[0] * sx, g[1] * sy), (g[2] - g[0]) * sx,
                                        (g[3] - g[1]) * sy, fill=False, edgecolor="#1a9850",
                                        linewidth=4.5, linestyle=(0, (4, 2)), zorder=2))
    n_pred = len(r["preds"])
    for j, p in enumerate(r["preds"] if all_boxes else r["preds"][:1]):
        ax.add_patch(mpatches.Rectangle((p[0] * sx, p[1] * sy), (p[2] - p[0]) * sx,
                                        (p[3] - p[1]) * sy, fill=False, edgecolor="#d73027",
                                        linewidth=1.6 if j == 0 else 0.9,
                                        linestyle="-" if j == 0 else ":", zorder=3))
    if not r["preds"]:
        plabel = "pred: none"
    elif all_boxes or n_pred == 1:
        plabel = f"pred x{n_pred} (first = solid)" if n_pred > 1 else "pred"
    else:
        plabel = f"pred (first of {n_pred}, scored)"
    ax.legend(handles=[
        mpatches.Patch(fill=False, edgecolor="#1a9850", linestyle="--", label=f"GT x{len(r['gts'])}"),
        mpatches.Patch(fill=False, edgecolor="#d73027", label=plabel)],
        loc="lower right", fontsize=9, framealpha=0.85)
    state = f"IoU {r['iou_first']:.2f}" if r["preds"] else "no valid box"
    ax.set_title(f"{r['class']}   {state}", fontsize=12)
    ax.axis("off")

    to100 = lambda b: [round(b[0] / W0 * 100, 1), round(b[1] / H0 * 100, 1),
                       round(b[2] / W0 * 100, 1), round(b[3] / H0 * 100, 1)]
    head = [f"image  {r['image_id']}", f"query  {r['class']}", f"size   {W0}x{H0} px",
            f"GT     (0-100) {[to100(g) for g in r['gts']]}",
            f"pred   (0-100) {[to100(p) for p in r['preds']] or 'none'}"]
    if r["oob"]:
        head.append(f"out-of-range (dropped): {[[round(v, 1) for v in b] for b in r['oob']]}")
    head += ["", "---- model answer ----"]
    at.axis("off")
    at.text(0.0, 1.0, "\n".join(head).replace("$", r"\$") + "\n" + _wrap(r["text"] or "(empty)"),
            va="top", ha="left", family="monospace", fontsize=8, transform=at.transAxes)
    fig.tight_layout()
    slug = "".join(c if c.isalnum() else "_" for c in r["class"])
    path = os.path.join(out_dir, f"{k:03d}_{slug}_{r['image_id'][:8]}_iou{r['iou_first']:.2f}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def scatter(rows, path):
    paired = [r for r in rows if "pred_c" in r]
    if len(paired) < 3:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    for ax, i, name in zip(axes, (0, 1), ("x center (0 = image left)", "y center (0 = image top)")):
        gx = [r["gt_c"][i] for r in paired]
        px = [r["pred_c"][i] for r in paired]
        ax.scatter(gx, px, s=9, alpha=0.35, color="#4575b4")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y = x (perfect)")
        k, b = V._linfit(gx, px)
        if not np.isnan(k):
            xs = np.array([0.0, 1.0])
            ax.plot(xs, k * xs + b, color="#d73027", linewidth=2, label=f"fit k={k:.2f}  b={b:+.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel(f"GT {name}")
        ax.set_ylabel("predicted")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"pred vs GT box center, n={len(paired)}   (Track 8 reference: x k=0.37, y k=0.44)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Visualize CheXOne VinDr results from cache")
    ap.add_argument("--data-root", default=V.DEFAULT_ROOT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--png-dir", default="./png_vindr")
    ap.add_argument("--cache-dir", default="./cache_vindr")
    ap.add_argument("--out", default="./vindr_result",
                    help="result dir of the run; figures go to <out>/vis/")
    V.add_run_args(ap)      # --runner/--pixels/--prompt/--reasoning/--max-new-tokens:
                            # all of them are part of the cache key, so they must match the run
    ap.add_argument("--pick", default="per-class", choices=PICKS,
                    help="per-class: --n per class | random/best/miss/nobox: --n in total")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--class", dest="vis_class", default=None, help="only this class name")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-boxes", action="store_true",
                    help="also draw boxes 2..n (dotted). Default: only the first, scored box")
    args = V.resolve_run_args(ap.parse_args())

    csv_path = args.csv or os.path.join(args.data_root, "physio", f"annotations_{args.split}.csv")
    pairs = V.load_pairs(csv_path)
    rows, miss_cache, miss_png = load_rows(args, pairs)
    print(f"[vis] {V.config_line(args)}: {len(rows)} pairs with cached answers "
          f"(no cache: {miss_cache}, no png: {miss_png})")
    if not rows:
        print("[vis] nothing to draw. The run flags (--runner/--pixels/--prompt/--max-new-tokens) "
              "must match the run, and so must --png-dir / --cache-dir.")
        return 1

    out_dir = os.path.join(args.out, "vis")
    os.makedirs(out_dir, exist_ok=True)
    if scatter(rows, os.path.join(out_dir, "scatter.png")):
        print(f"[vis] wrote {out_dir}/scatter.png")
    chosen = pick(rows, args.pick, args.n, args.seed)
    for k, r in enumerate(chosen):
        draw(r, k, out_dir, all_boxes=args.all_boxes)
    nb = sum(1 for r in rows if not r["preds"])
    print(f"[vis] pick={args.pick} n={args.n}: wrote {len(chosen)} figures to {out_dir}")
    print(f"[vis] all {len(rows)} pairs: no valid box {nb} ({100.0 * nb / len(rows):.0f}%), "
          f"IoU>0 {sum(r['iou_first'] > 0 for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
