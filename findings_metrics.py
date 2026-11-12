"""Report-generation metrics for findings_chexone.py output. No RadEval, no network.

  BLEU       corpus BLEU-1..4 (Papineni) + per-report BLEU-2 averaged over studies.
             The per-report BLEU-2 mean is what CXR-Report-Metric / ReXRank call "BLEU".
  ROUGE-L    LCS F-measure, beta 1.2 (pycocoevalcap convention), averaged over studies.
  BertScore  bert-score algorithm re-implemented on a LOCAL model (greedy token matching on
             hidden layer --num-layers, special tokens excluded, no idf), reported raw and
             baseline-rescaled with bert-score's official baseline for that model/layer.

Comparability with ReXRank (CheXOne-R1: BLEU 0.215, BertScore 0.441):
  * BLEU: same definition (BLEU-2 per report); tokenisation (lower-case words + punctuation)
    may differ slightly from CXR-Report-Metric.
  * BertScore: ReXRank uses distilroberta-base (rescaled). With distilbert-base-uncased the
    number is a different scale -> NOT directly comparable. Point --bert-model at a local
    distilroberta-base to compare.
  * SembScore / RadGraph / RaTEScore / RadCliQ / GREEN need their own models: n/a here.

  python findings_metrics.py --run-dir paper_suite/findings \
      --bert-model /mnt/data/jinghansun/Huggingface/Bert/distilbert/distilbert-base-uncased
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter

REFERENCE = {"1/RadCliQ-v1": 1.023, "BLEU": 0.215, "BertScore": 0.441, "SembScore": 0.456,
             "RadGraph": 0.234, "RaTEScore": 0.519, "GREEN": 0.314}
DEFAULT_BERT = "/mnt/data/jinghansun/Huggingface/Bert/distilbert/distilbert-base-uncased"
# bert-score rescale baselines (lang=en, no idf), F column, from bert_score/rescale_baseline/en/*.tsv
BASELINE_F = {
    ("distilbert-base-uncased", 5): 0.66620487,
    ("distilroberta-base", 5): 0.8473319,
}
DEFAULT_LAYERS = {"distilbert-base-uncased": 5, "distilroberta-base": 5}


# ------------------------------------------------------------------ text metrics
def tokens(s):
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", (s or "").lower())


def ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def sentence_bleu(ref, hyp, n_max=2):
    """Unsmoothed BLEU-n of one hypothesis vs one reference (0 if any n-gram precision is 0)."""
    if not hyp:
        return 0.0
    logs = []
    for n in range(1, n_max + 1):
        h = ngrams(hyp, n)
        tot = sum(h.values())
        hit = sum((h & ngrams(ref, n)).values())
        if tot == 0 or hit == 0:
            return 0.0
        logs.append(math.log(hit / tot))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return bp * math.exp(sum(logs) / n_max)


def corpus_bleu(refs, hyps, n_max=4):
    hit, tot = [0] * n_max, [0] * n_max
    rl = hl = 0
    for r, h in zip(refs, hyps):
        rl += len(r)
        hl += len(h)
        for n in range(1, n_max + 1):
            hn = ngrams(h, n)
            tot[n - 1] += sum(hn.values())
            hit[n - 1] += sum((hn & ngrams(r, n)).values())
    bp = 1.0 if hl > rl else (math.exp(1 - rl / hl) if hl else 0.0)
    out = {}
    for k in range(1, n_max + 1):
        if min(hit[:k]) == 0:
            out[f"BLEU-{k}"] = 0.0
        else:
            out[f"BLEU-{k}"] = bp * math.exp(sum(math.log(hit[i] / tot[i]) for i in range(k)) / k)
    return out


def rouge_l(ref, hyp, beta=1.2):
    if not ref or not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for a in ref:                               # LCS, O(len(ref) * len(hyp)) memory O(len(hyp))
        cur = [0]
        for j, b in enumerate(hyp):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return (1 + beta ** 2) * p * r / (r + beta ** 2 * p)


# ------------------------------------------------------------------ BERTScore (local model)
def bertscore(refs, hyps, model_path, num_layers, batch_size, device):
    """bert-score's greedy matching: P = mean_h max_r cos, R = mean_r max_h cos, F = 2PR/(P+R)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
    special = set(tok.all_special_ids)

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            b = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
            b = {k: v.to(device) for k, v in b.items()}
            with torch.no_grad():
                hs = model(**b, output_hidden_states=True).hidden_states[num_layers]
            hs = torch.nn.functional.normalize(hs, dim=-1)
            for j in range(hs.shape[0]):
                ids = b["input_ids"][j]
                keep = (b["attention_mask"][j] == 1) & torch.tensor([int(t) not in special for t in ids],
                                                                    device=ids.device)
                out.append(hs[j][keep])
        return out

    R, H = embed(refs), embed(hyps)
    P_, R_, F_ = [], [], []
    for r, h in zip(R, H):
        if len(r) == 0 or len(h) == 0:
            P_.append(0.0); R_.append(0.0); F_.append(0.0)  # noqa: E702
            continue
        sim = h @ r.T
        p = sim.max(dim=1).values.mean().item()
        rc = sim.max(dim=0).values.mean().item()
        P_.append(p); R_.append(rc); F_.append(2 * p * rc / (p + rc) if p + rc > 0 else 0.0)  # noqa: E702
    return P_, R_, F_


def read_csv(p):
    with open(p, "r", encoding="utf-8") as f:
        return {r["study_id"]: r["report"] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="output dir of findings_chexone.py")
    ap.add_argument("--gt", default="findings", choices=["findings", "report"],
                    help="findings: section_findings (the task). report: 'Findings: .. Impression: ..'")
    ap.add_argument("--bert-model", default=DEFAULT_BERT, help="local HF model dir for BertScore ('' = skip)")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="hidden layer used by BertScore (default: bert-score's choice for the model, 5)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pred = read_csv(os.path.join(args.run_dir, "pred.csv"))
    if args.gt == "findings":
        gt = read_csv(os.path.join(args.run_dir, "gt.csv"))
    else:
        with open(os.path.join(args.run_dir, "predictions.json"), "r", encoding="utf-8") as f:
            pj = json.load(f)
        if not any(v.get("section_impression") for v in pj.values()):
            print("[ERROR] --gt report needs section_impression; only the ReXRank json has it. Use --gt findings.")
            return 1
        gt = {k: f"Findings: {v.get('section_findings')} Impression: {v.get('section_impression')}"
              for k, v in pj.items()}
    ids = sorted(set(pred) & set(gt))
    empty = sum(1 for i in ids if not pred[i].strip())
    refs = [" ".join(gt[i].split()) for i in ids]
    hyps = [" ".join(pred[i].split()) for i in ids]
    print(f"[metrics] {len(ids)} studies (empty predictions: {empty}), reference = {args.gt}")

    rt, ht = [tokens(s) for s in refs], [tokens(s) for s in hyps]
    per = {"BLEU-2": [sentence_bleu(r, h, 2) for r, h in zip(rt, ht)],
           "ROUGE-L": [rouge_l(r, h) for r, h in zip(rt, ht)]}
    cb = corpus_bleu(rt, ht, 4)

    bs_info = None
    if args.bert_model:
        name = os.path.basename(os.path.normpath(args.bert_model))
        layers = args.num_layers or DEFAULT_LAYERS.get(name, 5)
        base = BASELINE_F.get((name, layers))
        try:
            import torch
            dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[metrics] BertScore: {args.bert_model}  layer {layers}  device {dev}")
            _, _, F = bertscore(refs, hyps, args.bert_model, layers, args.batch_size, dev)
            per["BertScore_raw"] = F
            if base is not None:
                per["BertScore"] = [(f - base) / (1 - base) for f in F]
            bs_info = {"model": name, "layer": layers, "baseline_F": base}
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] BertScore: {type(e).__name__}: {e}")

    mean = {k: sum(v) / len(v) for k, v in per.items() if v}
    mapped = {k: None for k in REFERENCE}
    mapped["BLEU"] = mean.get("BLEU-2")
    mapped["BertScore"] = mean.get("BertScore")

    print("\n" + "=" * 72)
    print(f"{'metric':<34s}{'ours':>10s}{'ReXRank':>12s}")
    print("-" * 72)
    rows = [("BLEU  (BLEU-2 per report, mean)", mean.get("BLEU-2"), REFERENCE["BLEU"])]
    if bs_info:
        lab = f"BertScore ({bs_info['model']}, rescaled)"
        rows.append((lab, mean.get("BertScore"), REFERENCE["BertScore"]))
        rows.append((f"BertScore ({bs_info['model']}, raw)", mean.get("BertScore_raw"), None))
    rows.append(("ROUGE-L (mean)", mean.get("ROUGE-L"), None))
    rows += [(k + " (corpus)", v, None) for k, v in cb.items()]
    for lab, v, ref in rows:
        print(f"{lab:<34s}{('n/a' if v is None else f'{v:.3f}'):>10s}{('' if ref is None else f'{ref:.3f}'):>12s}")
    print("=" * 72)
    if bs_info and bs_info["model"] != "distilroberta-base":
        print(f"NOTE: ReXRank's BertScore uses distilroberta-base; {bs_info['model']} gives a different "
              "scale, so compare BertScore only between our own runs.")
    if bs_info and bs_info["baseline_F"] is None:
        print("NOTE: no rescale baseline for this model/layer; only raw BertScore reported.")
    print("SembScore / RadGraph / RaTEScore / RadCliQ / GREEN: not computed (need their own models).")

    with open(os.path.join(args.run_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"gt": args.gt, "n": len(ids), "empty_pred": empty, "mapped": mapped, "means": mean,
                   "corpus_bleu": cb, "bertscore": bs_info, "reference": REFERENCE}, f, indent=2)
    with open(os.path.join(args.run_dir, "per_study_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(per)
        w.writerow(["study_id"] + cols)
        for i, sid in enumerate(ids):
            w.writerow([sid] + [f"{per[c][i]:.4f}" for c in cols])
    print(f"-> {args.run_dir}/summary_metrics.json, per_study_metrics.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
