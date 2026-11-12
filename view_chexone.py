"""CheXOne View Classification on MIMIC-CXR (paper Table 1: 900 images; CheXOne 0.9978, 95% CI 0.978-1.0).

Paper §4.3: "Given a single CXR, the model is tasked with identifying the imaging projection. We
construct the test set using the MIMIC-CXR dataset by randomly sampling 900 CXRs, uniformly
distributed across anteroposterior (AP), posteroanterior (PA), and lateral views. Accordingly, each
instruction is associated with three multiple-choice options corresponding to these views."

Our construction (the exact 900 images are not released):
  * official MIMIC-CXR **test** split (mimic-cxr-2.0.0-split.csv), labels from ViewPosition in
    mimic-cxr-2.0.0-metadata.csv
  * 300 AP + 300 PA + 300 lateral, seeded random sample. Lateral = ViewPosition LATERAL or LL
    (--lateral-codes to change)
  * three options, order shuffled per image (seeded) so a letter bias cannot pass for skill
  * option layout CheXOne was trained on: "Options:\n(A) xx\n(B) xx\n(C) xx"
    (grpo_prepare/1_prepare_data/prepare_vqa.py), reasoning suffix, answer = letter in \\boxed{}
Model code = chexone_official.OfficialRunner (README Quick Start).
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared  # noqa: E402,F401
from progress import pbar               # noqa: E402
from parse_utils import extract_boxed   # noqa: E402

PAPER = {"acc": 0.9978, "ci": (0.978, 1.0), "n": 900}
CLASSES = ["AP", "PA", "Lateral"]
_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPTS = {
    "mc_reason_paren": "What is the view of this chest X-ray?\nOptions:\n{opts}\n" + _SUFFIX,
    "mc_reason": "What is the view of this chest X-ray?\n{opts}\n" + _SUFFIX,
    "mc": "What is the view of this chest X-ray?\nOptions:\n{opts}\nAnswer with the option's letter.",
}
WORDS = {"AP": [r"\bAP\b", r"antero-?posterior"], "PA": [r"\bPA\b", r"postero-?anterior"],
         "Lateral": [r"\blateral\b", r"\bLL\b"]}


def _open(p):
    if os.path.exists(p):
        return open(p, "r", encoding="utf-8")
    if os.path.exists(p + ".gz"):
        return io.TextIOWrapper(gzip.open(p + ".gz"), encoding="utf-8")
    raise FileNotFoundError(f"{p}(.gz)")


def build_samples(meta_csv, split_csv, jpg_root, n_per_class, seed, lateral_codes):
    with _open(split_csv) as f:
        test = {r["dicom_id"] for r in csv.DictReader(f) if r["split"] == "test"}
    pools = defaultdict(list)
    with _open(meta_csv) as f:
        for r in csv.DictReader(f):
            if r["dicom_id"] not in test:
                continue
            vp = (r["ViewPosition"] or "").strip().upper()
            lab = "AP" if vp == "AP" else "PA" if vp == "PA" else "Lateral" if vp in lateral_codes else None
            if lab:
                s = r["subject_id"]
                p = os.path.join(jpg_root, f"p{s[:2]}", f"p{s}", f"s{r['study_id']}", f"{r['dicom_id']}.jpg")
                pools[lab].append({"dicom_id": r["dicom_id"], "label": lab, "img": p, "vp": vp})
    samples = []
    for c in CLASSES:
        pool = sorted(pools[c], key=lambda x: x["dicom_id"])
        k = min(n_per_class, len(pool))
        samples += random.Random(f"{seed}|{c}").sample(pool, k)
    return samples, {c: len(pools[c]) for c in CLASSES}


def build_prompt(s, prompt_name, seed):
    order = list(CLASSES)
    random.Random(f"{seed}|{s['dicom_id']}").shuffle(order)
    letters = "ABC"
    fmt = "({}) {}" if "paren" in prompt_name or prompt_name == "mc" else "{}. {}"
    opts = "\n".join(fmt.format(letters[i], o) for i, o in enumerate(order))
    return PROMPTS[prompt_name].replace("{opts}", opts), dict(zip(letters, order))


_LET = re.compile(r"^\W*\(?([ABC])\b")
_LET_ANY = re.compile(r"\b(?:answer|option)\s*(?:is|:)?\s*\(?([ABC])\b", re.I)


def parse_answer(text, letter_map):
    """-> (pred_class | None, how)."""
    if not text or not text.strip():
        return None, "empty"
    boxed = extract_boxed(text)
    src = boxed if boxed else text.strip().splitlines()[-1]
    src = re.sub(r"\\text\{([^}]*)\}", r"\1", src)
    m = _LET.match(src) or _LET_ANY.search(src)
    if m:
        return letter_map[m.group(1).upper()], ("boxed_letter" if boxed else "letter")
    hits = {c: max((x.start() for p in WORDS[c] for x in re.finditer(p, src, re.I)), default=-1)
            for c in CLASSES}
    c, pos = max(hits.items(), key=lambda t: t[1])
    if pos >= 0:
        return c, ("boxed_word" if boxed else "word")
    return None, "unparsed"


def cache_path(cache_dir, s, prompt, args):
    fp = hashlib.md5(f"{prompt}|{s['img']}|official|{args.pixels}|{args.max_new_tokens}"
                     .encode("utf-8")).hexdigest()[:10]
    return os.path.join(cache_dir, f"{s['dicom_id']}__{fp}.txt")


def summarize(recs, out_dir, cfg):
    y = np.array([r["label"] for r in recs])
    p = np.array([r["pred"] or "none" for r in recs])
    ok = (y == p).astype(float)
    rng = np.random.default_rng(0)
    bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(1000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    per = {c: float(ok[y == c].mean()) for c in CLASSES if (y == c).any()}
    conf = {c: dict(Counter(p[y == c])) for c in CLASSES}
    hows = Counter(r["how"] for r in recs)
    print("=" * 80)
    print(f"View classification, MIMIC-CXR test: n={len(recs)}   config: {cfg}")
    print(f"accuracy {ok.mean():.4f}  95% CI [{lo:.3f}, {hi:.3f}]   "
          f"paper {PAPER['acc']:.4f} ({PAPER['ci'][0]:.3f}-{PAPER['ci'][1]:.3f}), n={PAPER['n']}")
    print("per class recall:", {c: round(v, 3) for c, v in per.items()})
    print("confusion (true -> predicted):")
    for c in CLASSES:
        print(f"  {c:<8s} {conf[c]}")
    print("answer parsed as:", dict(hows), f"  unparsed {np.mean(p == 'none'):.1%}")
    print("=" * 80)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "n": len(recs), "acc": float(ok.mean()), "ci95": [lo, hi],
                   "per_class_recall": per, "confusion": conf, "parsed_as": dict(hows),
                   "unparsed": float(np.mean(p == "none")), "paper": PAPER}, f, indent=2)
    with open(os.path.join(out_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    print(f"[view] -> {out_dir}/summary.json, records.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-root", default="/mnt/data/jinghansun/data/MIMIC-CXR/2.0.0",
                    help="MIMIC-CXR-JPG 2.0.0 dir: has files/ and mimic-cxr-2.0.0-{metadata,split}.csv(.gz)")
    ap.add_argument("--meta-csv", default=None)
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--model", default=os.environ.get("CHEXONE_MODEL", "StanfordAIMI/CheXOne"))
    ap.add_argument("--prompt", default="mc_reason_paren", choices=sorted(PROMPTS))
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--lateral-codes", default="LATERAL,LL")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pixels", default="rec", choices=["rec", "default"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--out", default="./view_result")
    ap.add_argument("--cache-dir", default="./cache_view")
    ap.add_argument("--limit", type=int, default=0, help="first N per class (0 = all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shard", type=int, default=1)
    ap.add_argument("--reparse-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--show-raw", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def locate(name):
        # csvs may sit next to files/ (2.0.0/) or one level up (MIMIC-CXR/), plain or .gz
        cands = [os.path.join(d, name) for d in (args.mimic_root, os.path.dirname(os.path.normpath(args.mimic_root)))]
        for c in cands:
            if os.path.exists(c) or os.path.exists(c + ".gz"):
                return c
        return cands[0]
    meta = args.meta_csv or locate("mimic-cxr-2.0.0-metadata.csv")
    split = args.split_csv or locate("mimic-cxr-2.0.0-split.csv")
    print(f"[view] metadata: {meta}(.gz)\n[view] split:    {split}(.gz)")
    jpg_root = os.path.join(args.mimic_root, "files")
    lat = {c.strip().upper() for c in args.lateral_codes.split(",") if c.strip()}
    samples, avail = build_samples(meta, split, jpg_root, args.n_per_class, args.seed, lat)
    if args.limit > 0:
        samples = [s for c in CLASSES for s in [x for x in samples if x["label"] == c][:args.limit]]
    cfg = (f"runner=official pixels={args.pixels} max_new_tokens={args.max_new_tokens} "
           f"prompt={args.prompt} lateral={sorted(lat)} n={len(samples)}")
    if args.check or args.dry_run:
        print(f"[view] test-split images available: {avail}  -> sampled {Counter(s['label'] for s in samples)}")
        print(f"[view] lateral ViewPosition codes in sample: {Counter(s['vp'] for s in samples if s['label'] == 'Lateral')}")
        if args.check:
            miss = [s["img"] for s in samples if not os.path.exists(s["img"])]
            print(f"[view] image files missing: {len(miss)}/{len(samples)}" + (f"  e.g. {miss[:3]}" if miss else ""))
            return 1 if miss else 0
        for s in samples[:2]:
            t, m = build_prompt(s, args.prompt, args.seed)
            print(f"\n--- {s['dicom_id']} label={s['label']} options={m}\n{t}")
        return 0

    my = samples[args.shard::args.n_shard] if args.n_shard > 1 else samples
    os.makedirs(args.cache_dir, exist_ok=True)
    runner, recs, shown = None, [], 0
    for s in pbar(my, total=len(my), desc=f"view shard {args.shard}"):
        prompt, lmap = build_prompt(s, args.prompt, args.seed)
        cp = cache_path(args.cache_dir, s, prompt, args)
        if os.path.exists(cp):
            with open(cp, "r", encoding="utf-8") as f:
                text = f.read()
        elif args.reparse_only:
            continue
        else:
            if runner is None:
                from chexone_official import OfficialRunner
                runner = OfficialRunner(args.model, pixels=args.pixels,
                                        max_new_tokens=args.max_new_tokens, flash_attn=args.flash_attn)
                runner.load()
            try:
                text = runner.chat(s["img"], prompt) or ""
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            with open(cp, "w", encoding="utf-8") as f:
                f.write(text)
        pred, how = parse_answer(text, lmap)
        recs.append({"dicom_id": s["dicom_id"], "label": s["label"], "vp": s["vp"], "pred": pred,
                     "how": how, "options": lmap, "tail": text[-300:],
                     "imgs": [s["img"]], "prompt": prompt, "answer": text})   # for suite_vis.py
        if shown < args.show_raw:
            shown += 1
            print(f"\n----- {s['dicom_id']} label={s['label']} -> {pred} ({how})\nPROMPT: {prompt}\nANSWER: {text}")
    if args.no_eval:
        print(f"[shard {args.shard}] done: {len(recs)} samples")
    elif recs:
        summarize(recs, args.out, cfg)
    else:
        print("[view] no answers (wrong cache settings?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
