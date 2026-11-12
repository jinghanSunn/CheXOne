#!/usr/bin/env python
"""Per-sample figures for the paper-suite parts: image(s) + ground truth + CheXOne prediction.

Reads <run-dir>/records.json written by view_chexone.py / cxrlt_chexone.py / ms_cxr_chexone.py /
findings_chexone.py (records need the fields imgs|png, prompt, answer; older records lack them ->
re-score from cache first, e.g. `bash run_paper_suite.sh vis` does that, no GPU).
Abnormality grounding (VinDr) has its own vindr_vis.py.

  part      left                          right
  view      the image                     GT view vs predicted view, options, raw answer
  longtail  all views of the study        disease, GT present/absent vs prediction, raw answer
  phrase    image, GT boxes (green dashed) + first predicted box (red)   phrase, IoU, raw answer
  findings  all views of the study        reference findings | CheXOne findings, raw answer tail

--pick mix (default) = half failures, half successes; wrong / right / random.
Findings has no per-study paper metric; for picking only we rank by word-overlap F1
(printed as "overlap F1", not a ReXRank metric).

  python suite_vis.py --part findings --run-dir paper_suite/findings --n 12
"""
import os
import re
import sys
import json
import random
import argparse
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.patches as mpatches      # noqa: E402
from PIL import Image                      # noqa: E402

PARTS = ("view", "longtail", "phrase", "findings")
OK, BAD, GT_C, PR_C = "#1a9850", "#d73027", "#1a9850", "#d73027"


# ------------------------------------------------------------------ helpers
def esc(s):
    return (s or "").replace("$", r"\$")


def wrap(text, width, head=None, tail=None):
    t = (text or "").strip()
    if head and tail and len(t) > head + tail + 20:
        t = t[:head] + "\n  [... %d chars ...]\n" % (len(t) - head - tail) + t[-tail:]
    out = []
    for para in t.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return esc("\n".join(out))


def load_img(path, max_side=640):
    try:
        im = Image.open(path).convert("L")
        im.thumbnail((max_side, max_side))
        return im, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {path}"


def words(s):
    return re.findall(r"[a-z]+", (s or "").lower())


def overlap_f1(a, b):
    ca, cb = Counter(words(a)), Counter(words(b))
    inter = sum((ca & cb).values())
    if not inter:
        return 0.0
    p, r = inter / sum(cb.values()), inter / sum(ca.values())
    return 2 * p * r / (p + r)


def score(part, r):
    """-> (is_correct, sort_key) used for picking."""
    if part in ("view", "longtail"):
        ok = r.get("pred") is not None and r["pred"] == r["label"]
        return ok, float(ok)
    if part == "phrase":
        v = r.get("iou_first", 0.0)
        return v >= 0.5, v
    v = overlap_f1(r.get("findings"), r.get("report"))
    r["_f1"] = v
    return None, v


def pick(part, recs, mode, n, seed):
    rnd = random.Random(seed)
    sc = [(score(part, r), r) for r in recs]
    if mode == "random":
        rnd.shuffle(recs)
        return recs[:n]
    if part == "findings":            # no correct/incorrect: lowest / highest overlap
        s = sorted(sc, key=lambda t: t[0][1])
        lo, hi = [r for _, r in s[:n]], [r for _, r in s[::-1][:n]]
        if mode == "wrong":
            return lo
        if mode == "right":
            return hi
        out, seen = [], set()
        for r in lo[:n - n // 2] + hi:            # lows first, then highs, no duplicates
            if id(r) not in seen and len(out) < n:
                seen.add(id(r))
                out.append(r)
        return out
    wrong = [r for (ok, _), r in sc if not ok]
    right = [r for (ok, _), r in sc if ok]
    rnd.shuffle(wrong)
    rnd.shuffle(right)
    if mode == "wrong":
        return wrong[:n]
    if mode == "right":
        return right[:n]
    k = min(len(wrong), n - n // 2)
    return wrong[:k] + right[:n - k]


def image_axes(fig, gs_row, paths, title_prefix=""):
    n = max(1, min(3, len(paths)))
    sub = gs_row.subgridspec(1, n, wspace=0.03)
    for i in range(n):
        ax = fig.add_subplot(sub[0, i])
        ax.axis("off")
        if i >= len(paths):
            continue
        im, err = load_img(paths[i])
        if im is None:
            ax.text(0.5, 0.5, "image not found\n" + textwrap.fill(err, 40), ha="center", va="center",
                    fontsize=7, color=BAD, transform=ax.transAxes)
        else:
            ax.imshow(im, cmap="gray")
            ax.set_title(f"{title_prefix}view {i + 1}/{len(paths)}", fontsize=8)


# ------------------------------------------------------------------ per-part figures
def fig_classif(part, r, k, out_dir):
    paths = r.get("imgs") or []
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.04)
    image_axes(fig, gs[0, 0], paths)
    at = fig.add_subplot(gs[0, 1])
    at.axis("off")
    if part == "view":
        gt, pr = r["label"], r.get("pred")
        head = [f"VIEW  dicom {r['dicom_id'][:18]}  (ViewPosition {r.get('vp')})",
                f"GT: {gt}      pred: {pr or 'unparsed'}"]
        tag = f"view_{gt}_pred-{pr or 'none'}_{r['dicom_id'][:8]}"
    else:
        name = lambda v: {1: "present", 0: "absent", None: "unparsed"}[v]   # noqa: E731
        gt, pr = r["label"], r.get("pred")
        head = [f"LONG-TAIL  {r['disease']}  study {r['study']}  views={r.get('n_views')}",
                f"GT: {name(gt)}      pred: {name(pr)}"]
        tag = f"lt_{r['disease'].replace(' ', '_')}_gt{gt}_pred{pr}_{r['study']}"
    ok = pr is not None and pr == gt
    at.text(0, 1, esc(head[0]), fontsize=10, fontweight="bold", va="top", transform=at.transAxes)
    at.text(0, 0.955, esc(head[1]) + ("   CORRECT" if ok else "   WRONG"), fontsize=12,
            fontweight="bold", color=OK if ok else BAD, va="top", transform=at.transAxes)
    body = ("QUESTION:\n" + wrap(r.get("prompt"), 78) + "\n\nANSWER (parsed as " + str(r.get("how"))
            + "):\n" + wrap(r.get("answer") or r.get("tail"), 78, head=900, tail=600))
    at.text(0, 0.89, body, fontsize=7.6, family="monospace", va="top", transform=at.transAxes)
    out = os.path.join(out_dir, f"{k:03d}_{'OK' if ok else 'XX'}_{tag}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_phrase(r, k, out_dir):
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    im, err = load_img(r["png"], max_side=1024)
    if im is None:
        ax.text(0.5, 0.5, err, ha="center", va="center", color=BAD, fontsize=8, transform=ax.transAxes)
    else:
        ax.imshow(im, cmap="gray")
        sx, sy = im.size[0] / float(r["W0"]), im.size[1] / float(r["H0"])
        for g in r.get("gts", []):
            ax.add_patch(mpatches.Rectangle((g[0] * sx, g[1] * sy), (g[2] - g[0]) * sx, (g[3] - g[1]) * sy,
                                            fill=False, edgecolor=GT_C, linewidth=4.0, linestyle=(0, (5, 3)),
                                            zorder=2))
        p = r.get("pred_box")
        if p:
            ax.add_patch(mpatches.Rectangle((p[0] * sx, p[1] * sy), (p[2] - p[0]) * sx, (p[3] - p[1]) * sy,
                                            fill=False, edgecolor=PR_C, linewidth=1.8, zorder=3))
        ax.legend(handles=[
            mpatches.Patch(fill=False, edgecolor=GT_C, linestyle="--", label=f"GT x{len(r.get('gts', []))}"),
            mpatches.Patch(fill=False, edgecolor=PR_C,
                           label=(f"pred (first of {r['n_pred']}, scored)" if p else "pred: none"))],
            loc="lower right", fontsize=9, framealpha=0.85)
    v = r.get("iou_first", 0.0)
    ax.set_title(esc(f"{r['class']}   IoU {v:.2f}"), fontsize=12, color=OK if v >= 0.5 else BAD)
    at = fig.add_subplot(gs[0, 1])
    at.axis("off")
    at.text(0, 1, wrap("PHRASE: " + r.get("phrase", ""), 70), fontsize=11, fontweight="bold", va="top",
            transform=at.transAxes)
    body = ("QUESTION:\n" + wrap(r.get("prompt"), 80) + "\n\nANSWER:\n"
            + wrap(r.get("answer"), 80, head=900, tail=700))
    at.text(0, 0.9, body, fontsize=7.6, family="monospace", va="top", transform=at.transAxes)
    out = os.path.join(out_dir, f"{k:03d}_iou{v:.2f}_{r['class'].replace(' ', '_')}_{r['image_id'][:8]}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_findings(r, k, out_dir):
    paths = r.get("imgs") or []
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.06)
    image_axes(fig, gs[0, 0], paths)
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.05)
    f1 = r.get("_f1", overlap_f1(r.get("findings"), r.get("report")))
    fig.suptitle(esc(f"FINDINGS  study {r['study_id']}   views={r.get('n_views')}   extracted as "
                     f"{r.get('how')}   overlap F1 {f1:.2f} (word overlap, picking only)"),
                 fontsize=11, fontweight="bold", y=0.995)
    for j, (title, txt, c) in enumerate((("GROUND TRUTH (reference findings)", r.get("findings"), GT_C),
                                         ("CHEXONE (extracted from \\boxed{})", r.get("report"), PR_C))):
        a = fig.add_subplot(sub[0, j])
        a.axis("off")
        a.text(0, 1, title, fontsize=11, fontweight="bold", color=c, va="top", transform=a.transAxes)
        flat = " ".join((txt or "").split()) or "(empty)"    # MIMIC reports carry hard line breaks
        a.text(0, 0.94, wrap(flat, 70), fontsize=9, va="top", transform=a.transAxes)
        if j == 1:
            tail = (r.get("answer") or r.get("tail") or "")[-450:]
            a.text(0, 0.30, "raw answer, last 450 chars:\n" + wrap(tail, 88), fontsize=7, color="#555555",
                   family="monospace", va="top", transform=a.transAxes)
    out = os.path.join(out_dir, f"{k:03d}_f1{f1:.2f}_{r['study_id']}.png")
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out


def confusion(recs, out_dir):
    labs = sorted({r["label"] for r in recs})
    cols = labs + ["unparsed"]
    m = [[sum(1 for r in recs if r["label"] == a and (r.get("pred") or "unparsed") == b) for b in cols]
         for a in labs]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.imshow(m, cmap="Blues")
    for i in range(len(labs)):
        for j in range(len(cols)):
            ax.text(j, i, m[i][j], ha="center", va="center", fontsize=11)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(labs)), labs)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("view classification: confusion matrix")
    p = os.path.join(out_dir, "confusion.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=PARTS)
    ap.add_argument("--run-dir", required=True, help="dir with records.json; figures -> <run-dir>/vis/")
    ap.add_argument("--pick", default="mix", choices=["mix", "wrong", "right", "random"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter", default=None,
                    help="keep records whose class/disease/label contains this text, e.g. 'Pneumoperitoneum'")
    args = ap.parse_args()

    rp = os.path.join(args.run_dir, "records.json")
    if not os.path.exists(rp):
        print(f"[vis] {rp} not found - run that part first")
        return 1
    with open(rp, "r", encoding="utf-8") as f:
        recs = json.load(f)
    need = "png" if args.part == "phrase" else "imgs"
    if recs and not any(need in r for r in recs):
        print(f"[vis] records.json has no '{need}' field (written before vis support). Re-score from "
              f"cache (no GPU), e.g. `bash run_paper_suite.sh vis {args.part}`, then retry.")
        return 1
    if args.filter:
        key = {"view": "label", "longtail": "disease", "phrase": "class", "findings": "study_id"}[args.part]
        recs = [r for r in recs if args.filter.lower() in str(r.get(key, "")).lower()]
    out_dir = os.path.join(args.run_dir, "vis")
    os.makedirs(out_dir, exist_ok=True)
    chosen = pick(args.part, list(recs), args.pick, args.n, args.seed)
    print(f"[vis] {args.part}: {len(recs)} records, drawing {len(chosen)} ({args.pick}) -> {out_dir}")
    for k, r in enumerate(chosen):
        if args.part in ("view", "longtail"):
            p = fig_classif(args.part, r, k, out_dir)
        elif args.part == "phrase":
            p = fig_phrase(r, k, out_dir)
        else:
            p = fig_findings(r, k, out_dir)
        print("  ", os.path.basename(p))
    if args.part == "view" and recs:
        print("  ", os.path.basename(confusion(recs, out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
