"""CheXOne Findings Generation on the MIMIC-CXR test set (paper Table 1: 2,347 studies).

Test set: ReXRank's public MIMIC-CXR test json (data/ReXRank_MIMICCXR_test.json, from
github.com/rajpurkarlab/ReXrank, 2347 studies, official MIMIC-CXR test split). The paper says
baseline numbers come from the ReXRank leaderboard "where CheXOne was evaluated under fair
experimental conditions", so this is the same test set the paper used.

Input: all images of the study (ReXRank `image_path`, 1-5 views; paper: "single or multiple
radiographic inputs"), or only `key_image_path` with --views key.
Prompt: the README Quick Start example is exactly this task:
  instruct  "Write an example findings section for the CXR."
  reason    the same + " Please reason step by step, and put your final answer within \\boxed{{}}."
            (README literal: a plain Python string, so the model sees \\boxed{{}})
The report is the content of the last \\boxed{...}; if the model never closes it, the text after the
last "\\boxed{" is used; if there is no \\boxed at all, the whole answer. How each answer was
extracted is counted, because it changes what the metrics see.

Metrics are a separate step (findings_metrics.py) so they can run in another environment.

Outputs (in --out):
  predictions.json   ReXRank format: {study_id: {..., "model_prediction": "<report>"}}
  gt.csv / pred.csv  study_id,report   (CXR-Report-Metric input format)
  records.json       per study: n_views, how, len, raw tail
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _shared  # noqa: E402,F401
from progress import pbar               # noqa: E402
from parse_utils import extract_boxed   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "data", "ReXRank_MIMICCXR_test.json")
DEFAULT_JPG = "/mnt/data/jinghansun/data/MIMIC-CXR/2.0.0/files"

_BASE = "Write an example findings section for the CXR."
PROMPTS = {
    "instruct": _BASE,
    "reason": _BASE + " Please reason step by step, and put your final answer within \\boxed{{}}.",
    "reason_single": _BASE + " Please reason step by step, and put your final answer within \\boxed{}.",
}

# ReXRank leaderboard, MIMIC-CXR test, "CheXOne-R1 (Stanford)". The paper does not print its MIMIC
# findings numbers in the main text (supplementary only); this is the public reference we have.
REFERENCE = {"source": "ReXRank leaderboard, CheXOne-R1, MIMIC-CXR test",
             "1/RadCliQ-v1": 1.023, "BLEU": 0.215, "BertScore": 0.441, "SembScore": 0.456,
             "RadGraph": 0.234, "RaTEScore": 0.519, "GREEN": 0.314}


def load_test(path):
    """Two accepted formats, both give the same 2347-study MIMIC-CXR test set:
    * ReXRank json  {id: {image_path:[...], key_image_path, section_findings, ...}}
    * R2Gen-style MIMIC annotation.json {"train":[...], "val":[...], "test":[{study_id,
      report, image_path:[one jpg]}, ...]} -- one row per image; rows grouped by study.
      Checked locally: same 2347 studies, identical image sets, report == ReXRank
      section_findings (2336 verbatim, 11 differ only in whitespace).
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = []
    if isinstance(d, dict) and isinstance(d.get("test"), list):
        by = {}
        for x in d["test"]:
            s = by.setdefault(str(x["study_id"]), {"images": [], "findings": x.get("report") or ""})
            for p in x["image_path"]:
                if p not in s["images"]:
                    s["images"].append(p)
        for sid in sorted(by):
            s = by[sid]
            # no view labels in this format: key image = first listed image
            out.append({"study_id": sid, "images": s["images"], "key_image": s["images"][0],
                        "findings": s["findings"], "raw": {"section_findings": s["findings"]}})
        return out
    for sid in sorted(d):
        x = d[sid]
        out.append({"study_id": sid, "images": list(x["image_path"]), "key_image": x["key_image_path"],
                    "findings": x.get("section_findings") or "", "raw": x})
    return out


def img_abs(jpg_root, rel):
    rel = rel.lstrip("/\\")
    if rel.startswith("files/"):
        rel = rel[len("files/"):]
    return os.path.join(jpg_root, rel)


def images_for(s, args):
    rels = [s["key_image"]] if args.views == "key" else s["images"][:args.max_views]
    return [img_abs(args.jpg_root, r) for r in rels]


def extract_report(text):
    """-> (report, how)."""
    if not text or not text.strip():
        return "", "empty"
    b = extract_boxed(text)
    if b is not None:
        how = "boxed"
    else:
        i = text.rfind("\\boxed{")
        if i >= 0:
            b, how = text[i + len("\\boxed{"):], "boxed_unclosed"
        else:
            b, how = text, "no_boxed"
    b = re.sub(r"\\text\{([^{}]*)\}", r"\1", b)
    b = b.replace("\\n", "\n").strip().strip("{}").strip()
    return b, how


def cache_path(cache_dir, s, prompt, imgs, args):
    key = f"{prompt}|{'|'.join(imgs)}|official|{args.pixels}|{args.max_new_tokens}"
    fp = hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    return os.path.join(cache_dir, f"{s['study_id']}__{fp}.txt")


def write_outputs(recs, studies, args, cfg):
    os.makedirs(args.out, exist_ok=True)
    by = {r["study_id"]: r for r in recs}
    pred_json = {}
    for s in studies:
        if s["study_id"] in by:
            x = dict(s["raw"])
            x["model_prediction"] = by[s["study_id"]]["report"]
            pred_json[s["study_id"]] = x
    with open(os.path.join(args.out, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(pred_json, f, ensure_ascii=False, indent=1, default=str)
    for name, field in (("gt.csv", "findings"), ("pred.csv", "report")):
        with open(os.path.join(args.out, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["study_id", "report"])
            for r in recs:
                w.writerow([r["study_id"], r[field]])
    with open(os.path.join(args.out, "records.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    hows = Counter(r["how"] for r in recs)
    lens = sorted(len(r["report"].split()) for r in recs) or [0]
    summary = {"config": cfg, "n": len(recs), "n_expected": len(studies), "extracted_as": dict(hows),
               "report_words_median": lens[len(lens) // 2],
               "gt_words_median": sorted(len(r["findings"].split()) for r in recs)[len(recs) // 2] if recs else 0,
               "reference": REFERENCE}
    with open(os.path.join(args.out, "summary_gen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("=" * 90)
    print(f"Findings generation, MIMIC-CXR test ({os.path.basename(args.test_json)}): "
          f"{len(recs)}/{len(studies)} studies")
    print(f"config: {cfg}")
    print(f"report extracted as: {dict(hows)}")
    print(f"median words: generated {summary['report_words_median']}  reference {summary['gt_words_median']}")
    if hows.get("no_boxed", 0) + hows.get("boxed_unclosed", 0) > 0.05 * max(1, len(recs)):
        print("  WARNING: >5% answers without a closed \\boxed{}: the whole reasoning text may be scored.")
    print(f"-> {args.out}/predictions.json, gt.csv, pred.csv   next: findings_metrics.py --run-dir {args.out}")
    print("=" * 90)


def main():
    ap = argparse.ArgumentParser(description="CheXOne findings generation on MIMIC-CXR test")
    ap.add_argument("--test-json", default=DEFAULT_JSON)
    ap.add_argument("--jpg-root", default=DEFAULT_JPG, help="MIMIC-CXR-JPG 'files' dir (has p10/ ...)")
    ap.add_argument("--model", default=os.environ.get("CHEXONE_MODEL", "StanfordAIMI/CheXOne"))
    ap.add_argument("--prompt", default="reason", choices=sorted(PROMPTS))
    ap.add_argument("--views", default="all", choices=["all", "key"])
    ap.add_argument("--max-views", type=int, default=5)
    ap.add_argument("--pixels", default="rec", choices=["rec", "default"])
    ap.add_argument("--max-new-tokens", type=int, default=2048, help="paper script: 2048")
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--out", default="./findings_result")
    ap.add_argument("--cache-dir", default="./cache_findings")
    ap.add_argument("--limit", type=int, default=0, help="first N studies (sorted by id); 0 = all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shard", type=int, default=1)
    ap.add_argument("--reparse-only", action="store_true")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--show-raw", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    studies = load_test(args.test_json)
    if args.limit > 0:
        studies = studies[:args.limit]
    prompt = PROMPTS[args.prompt]
    cfg = (f"runner=official pixels={args.pixels} max_new_tokens={args.max_new_tokens} "
           f"prompt={args.prompt} views={args.views}")
    if args.dry_run or args.check:
        nv = [len(images_for(s, args)) for s in studies]
        print(f"[findings] {len(studies)} studies, views per study {Counter(nv)}  ({cfg})")
        print(f"[findings] prompt: {prompt!r}")
        if args.check:
            miss = [p for s in studies for p in images_for(s, args) if not os.path.exists(p)]
            print(f"[findings] image files missing: {len(miss)}/{sum(nv)}" + (f"  e.g. {miss[:3]}" if miss else ""))
            return 1 if miss else 0
        return 0

    my = studies[args.shard::args.n_shard] if args.n_shard > 1 else studies
    os.makedirs(args.cache_dir, exist_ok=True)
    runner, recs, shown = None, [], 0
    for s in pbar(my, total=len(my), desc=f"findings shard {args.shard}"):
        imgs = images_for(s, args)
        cp = cache_path(args.cache_dir, s, prompt, imgs, args)
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
                text = runner.chat(imgs if len(imgs) > 1 else imgs[0], prompt) or ""
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue
            with open(cp, "w", encoding="utf-8") as f:
                f.write(text)
        report, how = extract_report(text)
        recs.append({"study_id": s["study_id"], "n_views": len(imgs), "how": how, "report": report,
                     "findings": s["findings"], "tail": text[-400:],
                     "imgs": imgs, "prompt": prompt, "answer": text})   # for suite_vis.py
        if shown < args.show_raw:
            shown += 1
            print(f"\n----- study {s['study_id']} views={len(imgs)} how={how}\nANSWER: {text}\n"
                  f"EXTRACTED: {report}\nREFERENCE: {s['findings']}")

    if args.no_eval:
        print(f"[shard {args.shard}] done: {len(recs)} studies")
    elif recs:
        write_outputs(recs, studies, args, cfg)
    else:
        print("[findings] no answers (wrong cache settings?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
