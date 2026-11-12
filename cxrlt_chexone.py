"""CheXOne on the paper's "Long-tail Disease Classification" benchmark (CXRLongtail, 750 samples).

What the paper says (arXiv 2604.00493, §4.3 / Fig. 3):
  750 studies, 150 per disease, 5 rare diseases: pneumoperitoneum, pneumomediastinum,
  subcutaneous emphysema, tortuous aorta, aortic calcification. "Given one or more CXRs together
  with the name of a long-tail disease, the model is tasked with determining whether the queried
  disease is present", two multiple-choice options (present / absent), metric = accuracy.
  CheXOne 0.816 (0.791-0.842), CheXagent 0.768, MedGemma 0.567. Reasoning mode by default.

Where the data comes from (inferred, not stated in the paper):
  The repo's test file is named `mimic-cxr-lt_test_reasoning_750.json`, and exactly these 5 diseases
  are the 5 findings that the MICCAI 2023 "mimic-cxr-lt" (PruneCXR) subset adds on top of
  MIMIC-CXR-JPG. That subset ships in PhysioNet "cxr-lt-iccv-workshop-cvamd" v2.0.0 as
  miccai-2023_mimic-cxr-lt/miccai2023_mimic-cxr-lt_labels_test.csv.  Images = MIMIC-CXR-JPG v2.0.0.

What is NOT known, so it is a flag here (defaults are guesses, every setting goes into the cache key):
  --pos-frac      positives per disease (default 0.5 -> 75 pos + 75 neg)
  --views         study = all views of the study in the csv (paper: "one or more CXRs"), one = 1 image
  --prompt        question / option wording
  --shuffle-options  randomise A/B order per sample (seeded), so a letter bias cannot look like skill

Model code = chexone_official.OfficialRunner (README Quick Start, verbatim).
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared  # noqa: E402,F401  (puts ../track8_detection_sized on sys.path)
from progress import pbar               # noqa: E402
from parse_utils import extract_boxed   # noqa: E402

DISEASES = {   # csv column (normalised match)  ->  phrase shown to the model (paper wording)
    "Pneumoperitoneum": "pneumoperitoneum",
    "Pneumomediastinum": "pneumomediastinum",
    "Subcutaneous Emphysema": "subcutaneous emphysema",
    "Tortuous Aorta": "tortuous aorta",
    "Calcification of the Aorta": "aortic calcification",
}
PAPER = {"acc": 0.816, "ci": (0.791, 0.842), "n": 750}

_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
PROMPTS = {
    # multiple choice, letter answer. {opts} = "A. ...\nB. ..."
    "mc_reason": "Is there {ph} in the given chest X-ray?\n{opts}\n" + _SUFFIX,
    "mc": "Is there {ph} in the given chest X-ray?\n{opts}\nAnswer with the option's letter.",
    # same question, README literal double braces
    "mc_reason_dbl": "Is there {ph} in the given chest X-ray?\n{opts}\n"
                     "Please reason step by step, and put your final answer within \\boxed{{}}.",
    # free yes/no
    "yn_reason": "Does this chest X-ray show {ph}? Answer yes or no. " + _SUFFIX,
    # Option layout CheXOne was trained on (grpo_prepare/1_prepare_data/prepare_vqa.py rewrites every
    # classification question to  "...Options:\n(A) xx\n(B) xx"  and the answer to "(A) xx").
    "mc_reason_paren": "Is there {ph} in the given chest X-ray?\nOptions:\n{opts}\n" + _SUFFIX,
}
PAREN_PROMPTS = {"mc_reason_paren"}   # these get "(A) Yes" instead of "A. Yes"
OPT_PRESENT, OPT_ABSENT = "Yes", "No"


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ------------------------------------------------------------------ data
def load_csv(path, jpg_root):
    """-> list of rows {img, study, labels{disease: 0/1}}; column names detected, not assumed."""
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"[cxrlt] empty csv: {path}")
    cols = list(rows[0].keys())
    ncol = {_norm(c): c for c in cols}
    lab_col = {}
    for d in DISEASES:
        c = ncol.get(_norm(d))
        if c is None:   # e.g. "Calcification of the Aorta" vs "Aortic Calcification"
            c = next((ncol[k] for k in ncol if all(w in k for w in _norm(d).split("of"))), None)
        lab_col[d] = c
    missing = [d for d, c in lab_col.items() if c is None]
    if missing:
        raise SystemExit(f"[cxrlt] label columns not found for {missing}. csv columns: {cols}")

    path_col = next((ncol[k] for k in ("path", "fpath", "imagepath", "filepath") if k in ncol), None)
    sid_c, stu_c, dic_c = ncol.get("subjectid"), ncol.get("studyid"), ncol.get("dicomid")

    def img_path(r):
        if path_col:
            p = r[path_col].lstrip("/")
            if p.startswith("files/") or p.startswith("files\\"):
                p = p[6:]
            if not p.endswith(".jpg"):
                p = os.path.splitext(p)[0] + ".jpg"
            return os.path.join(jpg_root, p)
        if sid_c and stu_c and dic_c:
            s = r[sid_c]
            return os.path.join(jpg_root, f"p{s[:2]}", f"p{s}", f"s{r[stu_c]}", f"{r[dic_c]}.jpg")
        raise SystemExit(f"[cxrlt] no image path column and no subject/study/dicom ids: {cols}")

    def study_of(r, p):
        if stu_c:
            return str(r[stu_c])
        m = re.search(r"[\\/]s(\d+)[\\/]", p)
        return m.group(1) if m else p

    nf_col = ncol.get("nofinding")
    out = []
    for r in rows:
        p = img_path(r)
        labs = {}
        for d, c in lab_col.items():
            v = (r[c] or "0").strip()
            labs[d] = 1 if v in ("1", "1.0", "True", "true") else 0
        nf = 1 if nf_col and (r[nf_col] or "0").strip() in ("1", "1.0", "True", "true") else 0
        out.append({"img": p, "study": study_of(r, p), "labels": labs, "nofind": nf})
    info = {"cols": cols, "label_cols": lab_col, "path_col": path_col}
    return out, info


def build_samples(rows, n_per_class, pos_frac, seed, views, max_views=3, neg_pool="random"):
    """-> list of {disease, study, imgs[], label}; one study at most once per disease.

    neg_pool: random    = any study whose label for this disease is 0 (report did not mention it)
              nofinding = only studies labelled "No Finding" (cleaner negatives; labels are
                          report-derived, so "0" often just means "not mentioned")
    """
    studies = defaultdict(lambda: {"imgs": [], "labels": defaultdict(int), "nofind": 1})
    for r in rows:
        s = studies[r["study"]]
        s["imgs"].append(r["img"])
        for d, v in r["labels"].items():
            s["labels"][d] = max(s["labels"][d], v)
        s["nofind"] = min(s["nofind"], r.get("nofind", 0))   # all images of the study must be normal
    keys = sorted(studies)
    samples = []
    for d in DISEASES:
        rnd = random.Random(f"{seed}|{d}")
        pos = [k for k in keys if studies[k]["labels"][d] == 1]
        neg = [k for k in keys if studies[k]["labels"][d] == 0
               and (neg_pool == "random" or studies[k]["nofind"] == 1)]
        n_pos = min(len(pos), int(round(n_per_class * pos_frac)))
        n_neg = min(len(neg), n_per_class - n_pos)
        pick = [(k, 1) for k in rnd.sample(pos, n_pos)] + [(k, 0) for k in rnd.sample(neg, n_neg)]
        for k, y in sorted(pick):
            imgs = sorted(studies[k]["imgs"])
            imgs = imgs[:1] if views == "one" else imgs[:max_views]
            samples.append({"disease": d, "study": k, "imgs": imgs, "label": y,
                            "n_pos_avail": len(pos)})
    return samples


# ------------------------------------------------------------------ prompt / parse
def options_for(sample, shuffle, seed, paren=False):
    """-> (opts_text, letter_of_present). paren=True -> "(A) Yes" (training layout), else "A. Yes"."""
    present_first = True
    if shuffle:
        present_first = random.Random(f"{seed}|{sample['disease']}|{sample['study']}").random() < 0.5
    fmt = "({}) {}" if paren else "{}. {}"
    first, second = (OPT_PRESENT, OPT_ABSENT) if present_first else (OPT_ABSENT, OPT_PRESENT)
    return fmt.format("A", first) + "\n" + fmt.format("B", second), ("A" if present_first else "B")


def build_prompt(sample, prompt_name, shuffle, seed):
    opts, present_letter = options_for(sample, shuffle, seed, paren=prompt_name in PAREN_PROMPTS)
    text = PROMPTS[prompt_name].replace("{ph}", DISEASES[sample["disease"]]).replace("{opts}", opts)
    return text, present_letter


_LETTER = re.compile(r"^\W*\(?([AB])\b")
_LETTER_ANY = re.compile(r"\b(?:answer|option)\s*(?:is|:)?\s*\(?([AB])\b", re.I)
_YES = re.compile(r"\b(yes|present)\b", re.I)
_NO = re.compile(r"\b(no|absent|not present)\b", re.I)


def parse_answer(text, present_letter):
    """-> (pred_present: 1/0/None, how)."""
    if not text:
        return None, "empty"
    boxed = extract_boxed(text)
    src = boxed if boxed else text.strip().splitlines()[-1] if text.strip() else ""
    src = re.sub(r"\\text\{([^}]*)\}", r"\1", src)
    m = _LETTER.match(src) or _LETTER_ANY.search(src)
    if m:
        return int(m.group(1).upper() == present_letter), ("boxed_letter" if boxed else "letter")
    ys, ns = [x.start() for x in _YES.finditer(src)], [x.start() for x in _NO.finditer(src)]
    if ys or ns:   # last word wins ("not yes ... no")
        return int((max(ys) if ys else -1) > (max(ns) if ns else -1)), ("boxed_word" if boxed else "word")
    return None, "unparsed"


# ------------------------------------------------------------------ cache
def cache_key(args):
    return (f"{args.prompt}|{args.runner_tag}|{args.views}|shuf{int(args.shuffle_options)}|"
            f"seed{args.seed}")


def cache_path(cache_dir, sample, prompt, args):
    fp = hashlib.md5(f"{prompt}|{'|'.join(sample['imgs'])}|{cache_key(args)}".encode()).hexdigest()[:10]
    d = re.sub(r"\W+", "_", sample["disease"])
    return os.path.join(cache_dir, f"{d}__{sample['study']}__{fp}.txt")


# ------------------------------------------------------------------ metrics
def summarize(recs, out_dir, cfg):
    def stats(sub):
        y = np.array([r["label"] for r in sub])
        p = np.array([-1 if r["pred"] is None else r["pred"] for r in sub])
        ok = (p == y)
        pos, neg = y == 1, y == 0
        return {"n": len(sub), "acc": float(ok.mean()),
                "sens": float(ok[pos].mean()) if pos.any() else float("nan"),
                "spec": float(ok[neg].mean()) if neg.any() else float("nan"),
                "pred_yes_rate": float((p == 1).mean()),
                "unparsed": float((p == -1).mean()), "n_pos": int(pos.sum())}

    def boot(sub, n=1000):
        ok = np.array([(r["pred"] == r["label"]) for r in sub], dtype=float)
        rng = np.random.default_rng(0)
        bs = [ok[rng.integers(0, len(ok), len(ok))].mean() for _ in range(n)]
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    allst = stats(recs)
    lo, hi = boot(recs)
    per = {d: stats([r for r in recs if r["disease"] == d]) for d in DISEASES
           if any(r["disease"] == d for r in recs)}
    print("=" * 96)
    print(f"config: {cfg}")
    print(f"{'disease':<28s}{'n':>5s}{'n_pos':>7s}{'acc':>8s}{'sens':>8s}{'spec':>8s}"
          f"{'pred_yes':>10s}{'unparsed':>10s}")
    for d, s in list(per.items()) + [("ALL", allst)]:
        print(f"{d:<28s}{s['n']:>5d}{s['n_pos']:>7d}{s['acc']:>8.3f}{s['sens']:>8.3f}{s['spec']:>8.3f}"
              f"{s['pred_yes_rate']:>10.2f}{s['unparsed']:>10.1%}")
    print(f"\naccuracy {allst['acc']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   "
          f"balanced acc {(allst['sens'] + allst['spec']) / 2:.3f}")
    print(f"paper: {PAPER['acc']:.3f} (95% CI {PAPER['ci'][0]:.3f}-{PAPER['ci'][1]:.3f}), n={PAPER['n']}")
    hows = defaultdict(int)
    for r in recs:
        hows[r["how"]] += 1
    print("answer parsed as:", dict(hows))
    if allst["unparsed"] > 0.05:
        print("  WARNING: >5% unparsed -> look at raw answers (--show-raw) before trusting accuracy")
    print("=" * 96)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "overall": allst, "ci95": [lo, hi], "per_disease": per,
                   "paper": PAPER, "parsed_as": dict(hows)}, f, indent=2)
    with open(os.path.join(out_dir, "records.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    print(f"[cxrlt] -> {out_dir}/summary.json, records.json")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True,
                    help="miccai2023_mimic-cxr-lt_labels_test.csv (PhysioNet cxr-lt v2.0.0)")
    ap.add_argument("--jpg-root", required=True,
                    help="MIMIC-CXR-JPG 2.0.0 'files' dir (contains p10/, p11/, ...)")
    ap.add_argument("--model", default=os.environ.get("CHEXONE_MODEL", "StanfordAIMI/CheXOne"))
    ap.add_argument("--out", default="./cxrlt_result")
    ap.add_argument("--cache-dir", default="./cache_cxrlt")
    ap.add_argument("--prompt", default="mc_reason", choices=sorted(PROMPTS))
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--pos-frac", type=float, default=0.5)
    ap.add_argument("--views", default="study", choices=["study", "one"])
    ap.add_argument("--neg-pool", default="random", choices=["random", "nofinding"],
                    help="where negatives come from: any label-0 study, or only 'No Finding' studies")
    ap.add_argument("--no-shuffle-options", dest="shuffle_options", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pixels", default="rec", choices=["rec", "default"])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N samples per disease (0=all)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shard", type=int, default=1)
    ap.add_argument("--reparse-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--show-raw", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.runner_tag = f"official|{args.pixels}|{args.max_new_tokens}"

    rows, info = load_csv(args.labels_csv, args.jpg_root)
    samples = build_samples(rows, args.n_per_class, args.pos_frac, args.seed, args.views,
                            neg_pool=args.neg_pool)
    if args.limit > 0:
        keep = []   # same pos/neg ratio as the full set, so a small run is still balanced
        by = defaultdict(lambda: {0: [], 1: []})
        for s in samples:
            by[s["disease"]][s["label"]].append(s)
        for d in by:
            k1 = min(len(by[d][1]), int(round(args.limit * args.pos_frac)))
            keep += by[d][1][:k1] + by[d][0][:args.limit - k1]
        samples = keep
    cfg = (f"runner=official pixels={args.pixels} max_new_tokens={args.max_new_tokens} "
           f"prompt={args.prompt} views={args.views} pos_frac={args.pos_frac} neg_pool={args.neg_pool} "
           f"shuffle_options={args.shuffle_options} n={len(samples)}")

    if args.check or args.dry_run:
        print(f"[cxrlt] csv rows {len(rows)}  columns: {info['cols'][:12]}{' ...' if len(info['cols']) > 12 else ''}")
        print(f"[cxrlt] label columns: {info['label_cols']}  path column: {info['path_col']}")
        print(f"[cxrlt] samples {len(samples)}  ({cfg})")
        for d in DISEASES:
            sub = [s for s in samples if s["disease"] == d]
            npos_av = sub[0]["n_pos_avail"] if sub else 0
            print(f"  {d:<28s} n={len(sub):<4d} pos={sum(s['label'] for s in sub):<4d} "
                  f"(positive studies available in csv: {npos_av})"
                  + ("   <-- fewer than requested" if npos_av < args.n_per_class * args.pos_frac else ""))
        nv = [len(s["imgs"]) for s in samples]
        print(f"  views per sample: min {min(nv)} max {max(nv)} mean {np.mean(nv):.2f}")
        if args.check:
            miss = [p for s in samples for p in s["imgs"] if not os.path.exists(p)]
            n_img = sum(nv)
            print(f"[cxrlt] image files missing: {len(miss)}/{n_img}" + (f"  e.g. {miss[:3]}" if miss else ""))
            return 1 if miss else 0
        for s in samples[:3]:
            t, pl = build_prompt(s, args.prompt, args.shuffle_options, args.seed)
            print(f"\n--- {s['disease']} study {s['study']} label={s['label']} (present = {pl})\n{t}")
        return 0

    my = samples[args.shard::args.n_shard] if args.n_shard > 1 else samples
    os.makedirs(args.cache_dir, exist_ok=True)
    runner = None
    recs, shown = [], 0
    for s in pbar(my, total=len(my), desc=f"cxrlt shard {args.shard}"):
        prompt, present_letter = build_prompt(s, args.prompt, args.shuffle_options, args.seed)
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
                text = runner.chat(s["imgs"] if len(s["imgs"]) > 1 else s["imgs"][0], prompt)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            with open(cp, "w", encoding="utf-8") as f:
                f.write(text)
        pred, how = parse_answer(text, present_letter)
        recs.append({"disease": s["disease"], "study": s["study"], "n_views": len(s["imgs"]),
                     "label": s["label"], "pred": pred, "how": how, "present_letter": present_letter,
                     "tail": text[-300:],
                     "imgs": list(s["imgs"]), "prompt": prompt, "answer": text})   # for suite_vis.py
        if shown < args.show_raw:
            shown += 1
            print(f"\n----- {s['disease']} label={s['label']} present={present_letter} -> pred={pred} ({how})\n"
                  f"PROMPT: {prompt}\nANSWER: {text}")

    if args.no_eval:
        print(f"[shard {args.shard}] done: {len(recs)} samples")
    elif recs:
        if len(recs) < len(samples):
            print(f"[cxrlt] WARNING: only {len(recs)}/{len(samples)} samples have answers")
        summarize(recs, args.out, cfg)
    else:
        print("[cxrlt] no answers found (wrong cache settings?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
