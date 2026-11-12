"""CheXOne abnormality grounding on the official VinDr-CXR test split.

Question this answers: does our inference chain reproduce the paper's number
(arXiv 2604.00493: mIoU 0.452, mAP 0.471, 743 samples) on the model's own home
dataset? If yes, the failure on Track 8 is distribution shift. If no, suspect
our preprocessing / prompt / decoding first.

Protocol (paper Sec 4.3): given a CXR and the name of an abnormality that IS
present, localize it. No yes/no gate, no absent-class queries.
Fixed to the training convention: short side 512, coordinates 0-100.

Known gaps versus the paper:
  * The 743 samples cannot be reproduced from annotations_test.csv: there are
    2108 (image, class) pairs, 1675 of them single-box, and no class subset we
    tried gives 743. We evaluate every pair and report subsets.
  * The paper does not define its mAP or IoU threshold. Without per-box scores,
    Hit@t equals precision at threshold t, so compare mIoU first.

Why not image_prep.dicom_to_png_file: accession_from_path takes the grandparent
directory, which is "Vin-DR" for every file here, so all PNGs would collide.

Usage: see run_vindr_eval.sh
"""
import os
import csv
import sys
import json
import hashlib
import argparse
import random as _random
from collections import defaultdict

import numpy as np

import _shared  # noqa: F401  (bootstrap sys.path)
from progress import pbar
from image_prep import dicom_to_png
from parse_utils import extract_grounding, to_pixel

DEFAULT_ROOT = "/mnt/data/jinghansun/data/Vin-DR"
DEFAULT_MODEL = (os.environ.get("CHEXONE_MODEL")
                 or "/mnt/data/jinghansun/Huggingface/StanfordAIMI/CheXOne")

SHORT_SIDE = 512
COORD_MODE = "norm100"
MIN_PIXELS_TOKENS, MAX_PIXELS_TOKENS = 256, 640   # 512x615 must not be re-shrunk

# The 14 classes kept by the Kaggle release; the common subset in other papers.
KAGGLE14 = {"Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
            "Consolidation", "ILD", "Infiltration", "Lung Opacity", "Nodule/Mass",
            "Other lesion", "Pleural effusion", "Pleural thickening", "Pneumothorax",
            "Pulmonary fibrosis"}

# Prompt variants. These are the only thing that changes between experiments; the model code
# is the official one (chexone_official.py). Sent to the model byte for byte: {ph} is swapped
# with str.replace, never str.format, so the braces in \boxed{} / \boxed{{}} mean what they say.
#
# The README and HF model card contain NO grounding prompt (their only example is report
# generation). What they do show is the reasoning-mode convention: the instruct sentence plus
# the fixed suffix  Please reason step by step, and put your final answer within \\boxed{{}}.
# written in a plain (non f-) Python string, so the model literally sees \boxed{{}}.
_SUFFIX_DBL = "Please reason step by step, and put your final answer within \\boxed{{}}."
_FORMAT_LINES = ("Output format:\n<|ref|>target name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>\n"
                 "Example: <|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>\n")
PROMPTS = {
    # Mirror's native instruct template.
    "instruct": "Detect {ph} in the given image.",
    # Mirror's native reasoning template, \boxed{} with single braces (what we ran so far).
    "reason": ("Detect {ph} in the given image.\n" + _FORMAT_LINES
               + "Please reason step by step, and put your final answer within \\boxed{}."),
    # Same, but with the README's literal suffix \boxed{{}}.
    "reason_dbl": "Detect {ph} in the given image.\n" + _FORMAT_LINES + _SUFFIX_DBL,
    # README convention applied to grounding: instruct sentence + literal suffix, no format lines.
    "suffix": "Detect {ph} in the given image. " + _SUFFIX_DBL,
}
HIT_THRS = (0.1, 0.25, 0.3, 0.5, 0.75)


def phrase(cls):
    return "ILD" if cls == "ILD" else cls.lower()


def build_prompt(cls, style):
    """style: a PROMPTS key. A bool is accepted for old callers (True = 'reason')."""
    if style is None or isinstance(style, bool):
        style = "reason" if style else "instruct"
    return PROMPTS[style].replace("{ph}", phrase(cls))


def add_run_args(ap):
    """Arguments that decide which cache entries a run reads and writes. Shared with vindr_vis."""
    ap.add_argument("--runner", default="official", choices=["official", "ours"],
                    help="official: README Quick Start code verbatim (chexone_official.py). "
                         "ours: local_runner.QwenLocalRunner as used for Qwen3.5")
    ap.add_argument("--pixels", default="rec", choices=["rec", "default"],
                    help="official runner only. rec: min_pixels=256*28*28, max_pixels=512*512 "
                         "(README recommendation, paper script MAX_PIXELS=262144). "
                         "default: bare processor, 4-16384 visual tokens")
    ap.add_argument("--prompt", default=None, choices=sorted(PROMPTS),
                    help="prompt variant; default instruct, or reason with --reasoning")
    ap.add_argument("--reasoning", action="store_true", help="shorthand for --prompt reason")
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="official runner. README uses 1024, the paper script 2048")


def resolve_run_args(args):
    args.prompt_name = args.prompt or ("reason" if args.reasoning else "instruct")
    return args


def cache_variant(args):
    """Extra cache-key part. Empty for the old runner so its existing caches stay valid."""
    if args.runner == "official":
        return f"official|{args.pixels}|{args.max_new_tokens}"
    return ""


def load_pairs(csv_path):
    """-> {(image_id, class_name): [[x1,y1,x2,y2], ...]} in original pixels."""
    pairs = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["class_name"] == "No finding" or not r["x_min"]:
                continue
            pairs[(r["image_id"], r["class_name"])].append(
                [float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")])
    return dict(pairs)


def select_images(pairs, limit, seed, shard, n_shard):
    ids = sorted({k[0] for k in pairs})
    if limit > 0:
        order = list(range(len(ids)))
        _random.Random(seed).shuffle(order)
        ids = [ids[i] for i in sorted(order[:limit])]
    return ids[shard::n_shard] if n_shard > 1 else ids


def find_dicom(root, split, image_id):
    for ext in (".dicom", ".dcm", ".DICOM", ".DCM"):
        p = os.path.join(root, split, image_id + ext)
        if os.path.exists(p):
            return p
    return None


def png_for(dcm_path, image_id, png_dir):
    png = os.path.join(png_dir, f"{image_id}.png")
    meta = os.path.join(png_dir, f"{image_id}.info.json")
    if os.path.exists(png) and os.path.exists(meta):
        with open(meta, "r", encoding="utf-8") as f:
            return png, json.load(f)
    data, info = dicom_to_png(dcm_path, short_side=SHORT_SIDE)
    info["acc"] = image_id
    with open(png, "wb") as f:
        f.write(data)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return png, info


def cache_path(cache_dir, image_id, cls, prompt, variant=""):
    key = f"{prompt}|{SHORT_SIDE}|{MIN_PIXELS_TOKENS}|{MAX_PIXELS_TOKENS}"
    if variant:
        key += f"|{variant}"
    fp = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
    slug = "".join(c if c.isalnum() else "_" for c in cls)
    return os.path.join(cache_dir, f"{image_id}__{slug}__{fp}.txt")


# ---------------------------------------------------------------- metrics
def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mask(boxes, W, H, s):
    m = np.zeros((max(1, int(np.ceil(H * s))), max(1, int(np.ceil(W * s)))), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[int(y1 * s):int(np.ceil(y2 * s)), int(x1 * s):int(np.ceil(x2 * s))] = True
    return m


def union_iou(preds, gts, W, H, grid=512):
    if not preds:
        return 0.0
    s = grid / float(max(W, H))
    a, b = _mask(preds, W, H, s), _mask(gts, W, H, s)
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def nearest_gt(p, gts):
    """Highest IoU; center distance breaks ties (all-zero IoU is the common case)."""
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return max(gts, key=lambda g: (iou(p, g),
                                   -((g[0] + g[2]) / 2 - cx) ** 2
                                   - ((g[1] + g[3]) / 2 - cy) ** 2))


def score_pair(image_id, cls, text, gts, info, stats):
    W0, H0 = info["W0"], info["H0"]
    raw = extract_grounding(text, stats=stats)
    if raw:
        stats["grounding_hit"] = stats.get("grounding_hit", 0) + 1
    preds = []
    for _, b in raw:
        px = to_pixel(b, COORD_MODE, info["W_in"], info["H_in"], W0, H0)
        if px is None:
            stats["bad_box"] = stats.get("bad_box", 0) + 1
        else:
            preds.append(px)
    rec = {"image_id": image_id, "class": cls, "n_gt": len(gts), "n_pred": len(preds),
           "iou_first": 0.0, "iou_union": union_iou(preds, gts, W0, H0), "no_box": not preds}
    if preds:
        g = nearest_gt(preds[0], gts)
        p = preds[0]
        rec.update(iou_first=iou(p, g),
                   pred_box=[round(v, 1) for v in p],
                   pred_c=[(p[0] + p[2]) / 2 / W0, (p[1] + p[3]) / 2 / H0],
                   gt_c=[(g[0] + g[2]) / 2 / W0, (g[1] + g[3]) / 2 / H0])
    return rec


def summarize(recs):
    v = np.array([r["iou_first"] for r in recs], dtype=float)
    u = np.array([r["iou_union"] for r in recs], dtype=float)
    out = {"n": len(recs), "mIoU": float(v.mean()), "medIoU": float(np.median(v)),
           "mIoU_union": float(u.mean()),
           "no_box": float(np.mean([r["no_box"] for r in recs]))}
    out.update({f"Hit@{t}": float((v >= t).mean()) for t in HIT_THRS})
    return out


def bootstrap_ci(recs, n_boot=1000, seed=0):
    """Percentile CI, resampling images because pairs in one image are correlated."""
    per = defaultdict(lambda: [0.0, 0])
    for r in recs:
        per[r["image_id"]][0] += r["iou_first"]
        per[r["image_id"]][1] += 1
    arr = np.array(list(per.values()), dtype=float)
    idx = np.random.RandomState(seed).randint(0, len(arr), (n_boot, len(arr)))
    means = arr[idx, 0].sum(1) / arr[idx, 1].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _linfit(xs, ys):
    x, y = np.array(xs, dtype=float), np.array(ys, dtype=float)
    if len(x) < 3 or x.std() < 1e-9:
        return float("nan"), float("nan")
    k, b = np.polyfit(x, y, 1)
    return float(k), float(b)


def report(recs, stats, out_dir, n_expected, cfg=""):
    cols = ["n", "mIoU", "medIoU", "mIoU_union", "Hit@0.1", "Hit@0.25", "Hit@0.3",
            "Hit@0.5", "Hit@0.75", "no_box"]

    def line(name, s):
        cells = "".join(f"{s[c]:>9d}" if c == "n" else f"{s[c]:>11.3f}" for c in cols)
        print(f"{name:<24s}{cells}")

    head = "".join(f"{c:>9s}" if c == "n" else f"{c:>11s}" for c in cols)
    print("\n" + "=" * 134)
    print(f"VinDr-CXR test / CheXOne grounding: {len(recs)}/{n_expected} (image, class) pairs "
          f"from {len({r['image_id'] for r in recs})} images")
    print("Paper (743 samples): mIoU 0.452 (95% CI 0.407-0.513)  mAP 0.471 (0.425-0.524)")
    print(f"config: {cfg}")
    print("=" * 134)
    print(f"{'subset':<24s}{head}")
    print("-" * 134)
    overall = summarize(recs)
    subsets = {"ALL": overall}
    line("ALL pairs", overall)
    for name, f in (("Kaggle 14 classes", lambda r: r["class"] in KAGGLE14),
                    ("Kaggle 14, no Other", lambda r: r["class"] in KAGGLE14
                     and r["class"] != "Other lesion"),
                    ("single-GT pairs", lambda r: r["n_gt"] == 1)):
        sub = [r for r in recs if f(r)]
        if sub:
            subsets[name] = summarize(sub)
            line(name, subsets[name])

    per = {}
    for cls in sorted({r["class"] for r in recs}):
        per[cls] = summarize([r for r in recs if r["class"] == cls])
    print("\n[per class, sorted by mIoU]")
    print(f"{'class':<24s}{head}")
    print("-" * 134)
    for cls in sorted(per, key=lambda c: -per[c]["mIoU"]):
        line(cls, per[cls])

    macro = {k: float(np.mean([p[k] for p in per.values()])) for k in overall if k != "n"}
    lo, hi = bootstrap_ci(recs)
    print(f"\nmIoU {overall['mIoU']:.3f}  95% CI [{lo:.3f}, {hi:.3f}] (image-level bootstrap)")
    print(f"macro over {len(per)} classes: mIoU {macro['mIoU']:.3f}  Hit@0.5 {macro['Hit@0.5']:.3f}")

    paired = [r for r in recs if "pred_c" in r]
    kx, bx = _linfit([r["gt_c"][0] for r in paired], [r["pred_c"][0] for r in paired])
    ky, by = _linfit([r["gt_c"][1] for r in paired], [r["pred_c"][1] for r in paired])
    print(f"\nregression pred_c = k*gt_c + b (n={len(paired)}):  "
          f"x: k={kx:.3f} b={bx:+.3f}   y: k={ky:.3f} b={by:+.3f}")
    print("Track 8 was x: k=0.369 b=+0.200   y: k=0.436 b=+0.202. k near 1 = real localization.")

    nq = max(1, len(recs))
    print(f"\nparse health: grounding hit {100.0 * stats.get('grounding_hit', 0) / nq:.1f}%  "
          f"bare tuple {stats.get('gsrc_text_tuple', 0) + stats.get('gsrc_boxed_tuple', 0)}  "
          f"out-of-range box {stats.get('bad_box', 0)}  "
          f"multi-box answers {sum(r['n_pred'] > 1 for r in recs)}")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "subsets": subsets, "macro": macro, "per_class": per,
                   "mIoU_ci95": [lo, hi], "regression": {"kx": kx, "bx": bx, "ky": ky, "by": by},
                   "stats": stats}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "per_class.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["class"] + cols)
        w.writeheader()
        for cls in sorted(per, key=lambda c: -per[c]["mIoU"]):
            w.writerow({"class": cls, **{c: per[cls][c] for c in cols}})
    with open(os.path.join(out_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    print(f"\nwritten: {out_dir}/summary.json, per_class.csv, records.json")


# ---------------------------------------------------------------- modes
def do_check(args, pairs):
    ids = sorted({k[0] for k in pairs})
    missing = [i for i in ids if not find_dicom(args.data_root, args.split, i)]
    print(f"[check] csv: {len(pairs)} pairs, {len(ids)} images; "
          f"missing DICOM under {args.split}/: {len(missing)}")
    if missing:
        print("        first missing:", missing[:5])
        return 1
    for image_id in ids[:3]:
        try:
            _, info = dicom_to_png(find_dicom(args.data_root, args.split, image_id),
                                   short_side=SHORT_SIDE)
        except Exception as e:  # noqa: BLE001
            print(f"[check] decode FAILED {image_id}: {type(e).__name__}: {e}")
            print("        likely compressed pixel data: pip install pylibjpeg "
                  "pylibjpeg-libjpeg python-gdcm")
            return 1
        gts = [b for k, v in pairs.items() if k[0] == image_id for b in v]
        mx, my = max(b[2] for b in gts), max(b[3] for b in gts)
        ok = mx <= info["W0"] and my <= info["H0"]
        print(f"[check] {image_id}: {info['W0']}x{info['H0']} -> {info['W_in']}x{info['H_in']}  "
              f"GT max ({mx:.0f},{my:.0f}) {'inside image' if ok else 'OUTSIDE IMAGE'}")
    if not os.path.isdir(args.model):
        print(f"[check] model dir not found: {args.model}")
        return 1
    print(f"[check] model dir ok: {args.model}\n[check] all good")
    return 0


def config_line(args):
    if args.runner == "official":
        return (f"runner=official  pixels={args.pixels}  prompt={args.prompt_name}  "
                f"max_new_tokens={args.max_new_tokens}")
    return f"runner=ours  prompt={args.prompt_name}  (min/max tokens {MIN_PIXELS_TOKENS}/{MAX_PIXELS_TOKENS})"


def run(args, pairs):
    images = select_images(pairs, args.limit, args.limit_seed, args.shard, args.n_shard)
    by_img = defaultdict(list)
    for (image_id, cls), gts in pairs.items():
        by_img[image_id].append((cls, gts))
    variant = cache_variant(args)

    runner = None
    if not args.reparse_only:
        if args.runner == "official":
            from chexone_official import OfficialRunner
            runner = OfficialRunner(args.model, pixels=args.pixels,
                                    max_new_tokens=args.max_new_tokens,
                                    flash_attn=args.flash_attn)
        else:
            from local_runner import QwenLocalRunner, GenerationConfig
            runner = QwenLocalRunner(
                model_name=args.model, flash_attn=args.flash_attn,
                min_pixels_tokens=MIN_PIXELS_TOKENS, max_pixels_tokens=MAX_PIXELS_TOKENS,
                torch_dtype=args.dtype,
                gen_config=GenerationConfig(
                    max_new_tokens=256 if args.prompt_name == "instruct" else 1024,
                    do_sample=False))
        runner.load()
        print(f"[vindr] {config_line(args)}")

    recs, stats, shown = [], {}, 0
    n_expected = sum(len(by_img[i]) for i in images)
    for image_id in pbar(images, total=len(images), desc="vindr"):
        dcm = find_dicom(args.data_root, args.split, image_id)
        try:
            if dcm is None and not os.path.exists(
                    os.path.join(args.png_dir, f"{image_id}.info.json")):
                raise FileNotFoundError("no DICOM")
            png, info = png_for(dcm, image_id, args.png_dir)
        except Exception as e:  # noqa: BLE001
            stats["image_fail"] = stats.get("image_fail", 0) + 1
            print(f"[WARN] {image_id}: {type(e).__name__}: {e}")
            continue
        for cls, gts in sorted(by_img[image_id]):
            prompt = build_prompt(cls, args.prompt_name)
            cp = cache_path(args.cache_dir, image_id, cls, prompt, variant)
            if os.path.exists(cp) and not args.overwrite_cache:
                with open(cp, "r", encoding="utf-8") as f:
                    text = f.read()
            elif args.reparse_only:
                stats["cache_miss"] = stats.get("cache_miss", 0) + 1
                continue
            else:
                text = runner.chat(png, prompt) or ""
                with open(cp, "w", encoding="utf-8") as f:
                    f.write(text)
            recs.append(score_pair(image_id, cls, text, gts, info, stats))
            if shown < args.show_raw:
                shown += 1
                print(f"\n--- {image_id} | {cls}\nGT: {[[round(v) for v in g] for g in gts]}"
                      f"\nW0xH0: {info['W0']}x{info['H0']}\nRAW: {text}\n"
                      f"IoU(first): {recs[-1]['iou_first']:.3f}")

    if stats.get("cache_miss"):
        print(f"[WARN] {stats['cache_miss']} pairs have no cached answer and are excluded")
    if args.no_eval:
        print(f"[shard {args.shard}] done: {len(recs)} pairs")
    elif recs:
        report(recs, stats, args.out, n_expected, config_line(args))
    else:
        print("[ERROR] nothing to evaluate")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="CheXOne grounding on VinDr-CXR")
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--csv", default=None,
                    help="default: {data-root}/physio/annotations_{split}.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--flash-attn", action="store_true")
    add_run_args(ap)
    ap.add_argument("--out", default="./vindr_result")
    ap.add_argument("--png-dir", default="./png_vindr")
    ap.add_argument("--cache-dir", default="./cache_vindr")
    ap.add_argument("--limit", type=int, default=0, help="images, not pairs; 0 = all")
    ap.add_argument("--limit-seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shard", type=int, default=1)
    ap.add_argument("--reparse-only", action="store_true")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--show-raw", type=int, default=0, help="print first N raw answers")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print prompts only")
    args = resolve_run_args(ap.parse_args())

    csv_path = args.csv or os.path.join(args.data_root, "physio", f"annotations_{args.split}.csv")
    pairs = load_pairs(csv_path)
    if args.dry_run:
        for cls in sorted({k[1] for k in pairs}):
            print(f"{cls:<22s} -> {build_prompt(cls, args.prompt_name)!r}")
        print(f"\n{len(pairs)} pairs, {len({k[0] for k in pairs})} images")
        print(f"config: {config_line(args)}")
        return 0
    if args.check:
        return do_check(args, pairs)
    for d in (args.out, args.png_dir, args.cache_dir):
        os.makedirs(d, exist_ok=True)
    return run(args, pairs)


if __name__ == "__main__":
    sys.exit(main())
