"""CheXOne phrase grounding on MS-CXR (paper: mIoU 0.608, mAP 0.835, 127 samples).

Same question as vindr_chexone.py, on the other grounding benchmark in the paper: does our
chain reproduce the number on data the model was trained on?

Task (Phrase Grounding): given a CXR and a free-text phrase describing an anatomical structure
or radiographic finding, localize the phrase to the corresponding region in the image.

Protocol: one query per (image, phrase). The query is the radiologist's free-text phrase from
the annotation ("Bibasilar opacities"), passed to the model verbatim (no lower-casing, no
category name), and scored against every box annotated for that phrase. Everything else (prompt
templates, model runner, parsing, IoU, metrics) is imported from vindr_chexone.py, so the two
benchmarks differ only in data.

Images: MIMIC-CXR-JPG (credentialed PhysioNet). MS-CXR only ships annotations. The csv `path`
column is relative to the MIMIC-CXR-JPG 2.0.0 root ("files/p10/p10233088/s.../x.jpg").

Known gaps versus the paper:
  * 127 samples cannot be reproduced. Test split: 216 boxes, 176 (image, finding, phrase)
    groups, 138 with one box, 126 with one box per (image, finding). Val split: 128 one-box
    groups. None equals 127. We evaluate whole splits and report the one-box subset.
  * The exact phrase-grounding instruction is not published. The only hint in the released
    code (MAIRA-2 baseline script) is that the user message looks like "<instruction>: <phrase>"
    and the answer carries <|ref|>phrase<|/ref|>. We reuse the templates of the grounding task
    in vindr_chexone.PROMPTS with the phrase in place of the class name.
  * The paper does not define its mAP. Without per-box scores Hit@t is precision at threshold t.

Usage: see run_msc_eval.sh
"""
import os
import io
import csv
import sys
import json
import zipfile
import hashlib
import argparse
from collections import defaultdict

import numpy as np

import vindr_chexone as V
from progress import pbar
from PIL import Image

DEFAULT_ANN = "/mnt/data/jinghansun/data/MS-CXR"
DEFAULT_JPG = "/mnt/data/jinghansun/data/MIMIC-CXR/2.0.0"
PAPER = "Paper (127 samples): mIoU 0.608  mAP 0.835"


# ---------------------------------------------------------------- annotations
def resolve_ann(path):
    """csv, zip containing the csv, or a directory containing either -> (kind, location)."""
    if os.path.isdir(path):
        csvs, zips = [], []
        for r, _, fs in os.walk(path):
            for f in fs:
                if f.endswith("v1.1.0.csv"):
                    csvs.append(os.path.join(r, f))
                elif f.endswith(".zip") and "ms-cxr" in f.lower():
                    zips.append(os.path.join(r, f))
        if csvs:
            return "csv", sorted(csvs)[0]
        if zips:
            return "zip", sorted(zips)[0]
        raise FileNotFoundError(f"no MS_CXR_Local_Alignment_v1.1.0.csv or zip under {path}")
    return ("zip" if path.endswith(".zip") else "csv"), path


def read_rows(path):
    kind, loc = resolve_ann(path)
    if kind == "zip":
        with zipfile.ZipFile(loc) as z:
            name = [n for n in z.namelist() if n.endswith("v1.1.0.csv")][0]
            text = z.read(name).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
    else:
        with open(loc, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    if rows and "split" not in rows[0]:
        raise ValueError(f"{loc} has no 'split' column: use the v1.1.0 release, not v1.0.0")
    return rows, loc


def ckey(cat, ph):
    """cache-name part: category + a short hash of the phrase (phrases can be long)."""
    return f"{cat}_{hashlib.md5(ph.encode('utf-8')).hexdigest()[:6]}"


def build_groups(rows, split):
    """-> list of {dicom_id, cat, ph, boxes(csv frame), path, cw, ch}.

    One group per (image, finding, phrase). The phrase is the query, verbatim. `cat` is the
    MS-CXR finding label and is only used for reporting, never shown to the model.
    """
    g = defaultdict(lambda: {"boxes": []})
    for r in rows:
        if split != "all" and r["split"] != split:
            continue
        ph = r["label_text"].strip()
        k = (r["dicom_id"], r["category_name"], ph)
        d = g[k]
        d.update(dicom_id=r["dicom_id"], cat=r["category_name"], ph=ph, path=r["path"],
                 cw=float(r["image_width"]), ch=float(r["image_height"]), split=r["split"])
        x, y, w, h = (float(r[c]) for c in ("x", "y", "w", "h"))
        d["boxes"].append([x, y, x + w, y + h])
    return [g[k] for k in sorted(g)]


# ---------------------------------------------------------------- images
def find_jpg(root, rel):
    cands = [os.path.join(root, rel)]
    if rel.startswith("files/"):
        cands.append(os.path.join(root, rel[len("files/"):]))
    else:
        cands.append(os.path.join(root, "files", rel))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def png_for_jpg(jpg, dicom_id, png_dir):
    """MIMIC-CXR-JPG -> PNG with short side 512 (same rule as image_prep.dicom_to_png)."""
    png = os.path.join(png_dir, f"{dicom_id}.png")
    meta = os.path.join(png_dir, f"{dicom_id}.info.json")
    if os.path.exists(png) and os.path.exists(meta):
        with open(meta, "r", encoding="utf-8") as f:
            return png, json.load(f)
    im = Image.open(jpg)
    W0, H0 = im.size
    im = im.convert("L")
    W_in, H_in = W0, H0
    if min(W0, H0) > V.SHORT_SIDE:
        s = float(V.SHORT_SIDE) / min(W0, H0)
        W_in, H_in = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
        im = im.resize((W_in, H_in), Image.BILINEAR)
    im.convert("RGB").save(png)
    info = {"W0": W0, "H0": H0, "W_in": W_in, "H_in": H_in, "acc": dicom_id}
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return png, info


def gt_in_image_frame(grp, info, stats):
    """Boxes in the csv are in the csv's image_width/height frame; the jpg should match."""
    sx, sy = info["W0"] / grp["cw"], info["H0"] / grp["ch"]
    if abs(sx - 1) > 0.01 or abs(sy - 1) > 0.01:
        stats["size_mismatch"] = stats.get("size_mismatch", 0) + 1
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in grp["boxes"]]


# ---------------------------------------------------------------- prompt
def build_prompt(ph, style):
    """The phrase goes in verbatim: it is free text, not a class name."""
    return V.PROMPTS[style].replace("{ph}", ph)


# ---------------------------------------------------------------- report
def _quartiles(recs):
    a = np.array([r["gt_area"] for r in recs])
    cuts = np.percentile(a, [25, 50, 75])
    bins = np.digitize(a, cuts)
    return [(f"Q{i + 1}", [r for r, b in zip(recs, bins) if b == i]) for i in range(4)]


def report(recs, stats, out_dir, n_expected, cfg, split):
    cols = ["n", "mIoU", "medIoU", "mIoU_union", "Hit@0.1", "Hit@0.25", "Hit@0.3",
            "Hit@0.5", "Hit@0.75", "no_box"]

    def line(name, s):
        cells = "".join(f"{s[c]:>9d}" if c == "n" else f"{s[c]:>11.3f}" for c in cols)
        print(f"{name:<24s}{cells}")

    head = "".join(f"{c:>9s}" if c == "n" else f"{c:>11s}" for c in cols)
    print("\n" + "=" * 134)
    print(f"MS-CXR {split} / CheXOne phrase grounding: {len(recs)}/{n_expected} (image, phrase) "
          f"queries from {len({r['image_id'] for r in recs})} images")
    print(PAPER)
    print(f"config: {cfg}")
    print("=" * 134)
    print(f"{'subset':<24s}{head}")
    print("-" * 134)
    overall = V.summarize(recs)
    subsets = {"ALL": overall}
    line("ALL queries", overall)
    one = [r for r in recs if r["n_gt"] == 1]
    if one:
        subsets["single-GT"] = V.summarize(one)
        line("single-GT queries", subsets["single-GT"])

    per = {c: V.summarize([r for r in recs if r["class"] == c])
           for c in sorted({r["class"] for r in recs})}
    print("\n[per finding, sorted by mIoU]")
    print(f"{'finding':<24s}{head}")
    print("-" * 134)
    for c in sorted(per, key=lambda c: -per[c]["mIoU"]):
        line(c, per[c])

    print("\n[by GT area, quartiles of box area / image area]")
    print(f"{'quartile':<24s}{head}")
    print("-" * 134)
    qs = {}
    for name, sub in _quartiles(recs):
        if sub:
            qs[name] = V.summarize(sub)
            ar = np.array([r["gt_area"] for r in sub])
            line(f"{name} ({ar.min():.1%}-{ar.max():.1%})", qs[name])

    macro = {k: float(np.mean([p[k] for p in per.values()])) for k in overall if k != "n"}
    lo, hi = V.bootstrap_ci(recs)
    print(f"\nmIoU {overall['mIoU']:.3f}  95% CI [{lo:.3f}, {hi:.3f}] (image-level bootstrap)")
    print(f"macro over {len(per)} findings: mIoU {macro['mIoU']:.3f}  Hit@0.5 {macro['Hit@0.5']:.3f}")

    paired = [r for r in recs if "pred_c" in r]
    kx, bx = V._linfit([r["gt_c"][0] for r in paired], [r["pred_c"][0] for r in paired])
    ky, by = V._linfit([r["gt_c"][1] for r in paired], [r["pred_c"][1] for r in paired])
    print(f"\nregression pred_c = k*gt_c + b (n={len(paired)}):  "
          f"x: k={kx:.3f} b={bx:+.3f}   y: k={ky:.3f} b={by:+.3f}")
    print("Reference: VinDr test x: k=0.60  y: k=0.68;  Track 8 x: k=0.37  y: k=0.44. "
          "k near 1 = real localization.")

    nq = max(1, len(recs))
    print(f"\nparse health: grounding hit {100.0 * stats.get('grounding_hit', 0) / nq:.1f}%  "
          f"bare tuple {stats.get('gsrc_text_tuple', 0) + stats.get('gsrc_boxed_tuple', 0)}  "
          f"out-of-range box {stats.get('bad_box', 0)}  "
          f"multi-box answers {sum(r['n_pred'] > 1 for r in recs)}  "
          f"jpg/csv size mismatch {stats.get('size_mismatch', 0)}")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "split": split, "subsets": subsets,
                   "macro": macro, "per_finding": per, "by_area": qs, "mIoU_ci95": [lo, hi],
                   "regression": {"kx": kx, "bx": bx, "ky": ky, "by": by}, "stats": stats},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "per_finding.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["finding"] + cols)
        w.writeheader()
        for c in sorted(per, key=lambda c: -per[c]["mIoU"]):
            w.writerow({"finding": c, **{k: per[c][k] for k in cols}})
    with open(os.path.join(out_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    print(f"\nwritten: {out_dir}/summary.json, per_finding.csv, records.json")


# ---------------------------------------------------------------- modes
def do_check(args, groups, ann_loc):
    imgs = sorted({g["dicom_id"] for g in groups})
    by_img = {g["dicom_id"]: g for g in groups}
    missing = [i for i in imgs if not find_jpg(args.jpg_root, by_img[i]["path"])]
    one = sum(len(g["boxes"]) == 1 for g in groups)
    print(f"[check] annotations: {ann_loc}")
    print(f"[check] split={args.split}: {len(groups)} (image, phrase) queries "
          f"({one} single-box), {len(imgs)} images; missing jpg under {args.jpg_root}: "
          f"{len(missing)}")
    if missing:
        print("        first missing:", [by_img[i]["path"] for i in missing[:3]])
        print("        jpg-root must be the MIMIC-CXR-JPG 2.0.0 directory that contains files/p10/...")
        return 1
    for i in imgs[:3]:
        g = by_img[i]
        im = Image.open(find_jpg(args.jpg_root, g["path"]))
        ok = abs(im.size[0] - g["cw"]) < 1 and abs(im.size[1] - g["ch"]) < 1
        print(f"[check] {i[:13]}: jpg {im.size[0]}x{im.size[1]}  csv {int(g['cw'])}x{int(g['ch'])}  "
              f"{'sizes match' if ok else 'SIZE MISMATCH (GT will be rescaled)'}")
    if not os.path.isdir(args.model):
        print(f"[check] model dir not found: {args.model}")
        return 1
    print(f"[check] model dir ok: {args.model}\n[check] all good")
    return 0


def run(args, groups):
    pairs = {(g["dicom_id"], g["cat"], g["ph"]): g for g in groups}
    images = V.select_images(pairs, args.limit, args.limit_seed, args.shard, args.n_shard)
    by_img = defaultdict(list)
    for g in groups:
        by_img[g["dicom_id"]].append(g)
    variant = V.cache_variant(args)

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
                min_pixels_tokens=V.MIN_PIXELS_TOKENS, max_pixels_tokens=V.MAX_PIXELS_TOKENS,
                torch_dtype=args.dtype,
                gen_config=GenerationConfig(
                    max_new_tokens=256 if args.prompt_name == "instruct" else 1024,
                    do_sample=False))
        runner.load()
        print(f"[msc] {V.config_line(args)}")

    recs, stats, shown = [], {}, 0
    n_expected = sum(len(by_img[i]) for i in images)
    for did in pbar(images, total=len(images), desc="ms-cxr"):
        first = by_img[did][0]
        try:
            jpg = find_jpg(args.jpg_root, first["path"])
            if jpg is None and not os.path.exists(os.path.join(args.png_dir, f"{did}.info.json")):
                raise FileNotFoundError(first["path"])
            png, info = png_for_jpg(jpg, did, args.png_dir)
        except Exception as e:  # noqa: BLE001
            stats["image_fail"] = stats.get("image_fail", 0) + 1
            print(f"[WARN] {did}: {type(e).__name__}: {e}")
            continue
        for g in by_img[did]:
            prompt = build_prompt(g["ph"], args.prompt_name)
            cp = V.cache_path(args.cache_dir, did, ckey(g["cat"], g["ph"]), prompt, variant)
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
            gts = gt_in_image_frame(g, info, stats)
            rec = V.score_pair(did, g["cat"], text, gts, info, stats)
            area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in gts) / float(info["W0"] * info["H0"])
            rec.update(phrase=g["ph"], split=g["split"], gt_area=area,
                       # for suite_vis.py: GT in original-image pixels, image actually fed
                       png=png, W0=info["W0"], H0=info["H0"],
                       gts=[[round(v, 1) for v in b] for b in gts], prompt=prompt, answer=text)
            recs.append(rec)
            if shown < args.show_raw:
                shown += 1
                print(f"\n--- {did[:13]} | {g['cat']} | {g['ph']!r}\n"
                      f"GT: {[[round(v) for v in b] for b in gts]}\n"
                      f"W0xH0: {info['W0']}x{info['H0']}\nRAW: {text}\n"
                      f"IoU(first): {rec['iou_first']:.3f}")

    if stats.get("cache_miss"):
        print(f"[WARN] {stats['cache_miss']} queries have no cached answer and are excluded")
    if args.no_eval:
        print(f"[shard {args.shard}] done: {len(recs)} queries")
    elif recs:
        report(recs, stats, args.out, n_expected, V.config_line(args), args.split)
    else:
        print("[ERROR] nothing to evaluate")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="CheXOne phrase grounding on MS-CXR")
    ap.add_argument("--ann", default=DEFAULT_ANN,
                    help="MS_CXR_Local_Alignment_v1.1.0.csv, the release zip, or a directory with either")
    ap.add_argument("--jpg-root", default=DEFAULT_JPG,
                    help="MIMIC-CXR-JPG 2.0.0 directory (contains files/p10/...)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--model", default=V.DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--flash-attn", action="store_true")
    V.add_run_args(ap)
    ap.add_argument("--out", default="./msc_result")
    ap.add_argument("--png-dir", default="./png_mscxr")
    ap.add_argument("--cache-dir", default="./cache_mscxr")
    ap.add_argument("--limit", type=int, default=0, help="images, not queries; 0 = all")
    ap.add_argument("--limit-seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shard", type=int, default=1)
    ap.add_argument("--reparse-only", action="store_true")
    ap.add_argument("--overwrite-cache", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--show-raw", type=int, default=0, help="print first N raw answers")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print prompts only")
    args = V.resolve_run_args(ap.parse_args())

    rows, ann_loc = read_rows(args.ann)
    groups = build_groups(rows, args.split)
    if args.dry_run:
        for g in groups[:12]:
            print(f"{g['cat']:<18s} -> {build_prompt(g['ph'], args.prompt_name)!r}")
        print(f"\n{len(groups)} queries ({sum(len(g['boxes']) == 1 for g in groups)} single-box), "
              f"{len({g['dicom_id'] for g in groups})} images, split={args.split}")
        print(f"config: {V.config_line(args)}")
        return 0
    if args.check:
        return do_check(args, groups, ann_loc)
    for d in (args.out, args.png_dir, args.cache_dir):
        os.makedirs(d, exist_ok=True)
    return run(args, groups)


if __name__ == "__main__":
    sys.exit(main())
