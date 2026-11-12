"""One table: our CheXOne re-run vs the paper (arXiv 2604.00493). Reads the summary files the
run_paper_suite.sh parts write; a missing part shows as "not run".

Writes <suite-dir>/paper_suite_summary.md and .json.
"""
import argparse
import json
import os
import sys


def _load(p):
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def in_ci(v, ci):
    if v is None or ci is None:
        return ""
    return "within paper CI" if ci[0] <= v <= ci[1] else ("above paper CI" if v > ci[1] else "below paper CI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", default="./paper_suite")
    a = ap.parse_args()
    d = a.suite_dir
    rows, notes = [], []

    s = _load(os.path.join(d, "view", "summary.json"))
    rows.append(("View classification", "MIMIC-CXR test", "Accuracy",
                 s and s["acc"], 0.9978, (0.978, 1.0), s and s["n"], 900,
                 s and f"unparsed {s['unparsed']:.1%}; recall {s['per_class_recall']}"))

    s = _load(os.path.join(d, "longtail", "summary.json"))
    o = s and s["overall"]
    rows.append(("Long-tail disease", "CXR-LT (MIMIC) test", "Accuracy",
                 o and o["acc"], 0.816, (0.791, 0.842), o and o["n"], 750,
                 o and f"sens {o['sens']:.3f} spec {o['spec']:.3f} pred_yes {o['pred_yes_rate']:.2f}; "
                       f"75 pos + 75 neg per disease (paper ratio unknown)"))

    for key, name, data, pm, pci, pn, pmap in (
            ("phrase", "Phrase grounding", "MS-CXR test", 0.608, (0.561, 0.660), 127, 0.835),
            ("abnormality", "Abnormality grounding", "VinDr-CXR test", 0.452, (0.407, 0.513), 743, 0.471)):
        s = _load(os.path.join(d, key, "summary.json"))
        al = s and s["subsets"]["ALL"]
        one = s and (s["subsets"].get("single-GT") or s["subsets"].get("single-GT pairs"))
        rows.append((name, data, "mIoU (per query)", al and al["mIoU"], pm, pci, al and al["n"], pn,
                     al and (f"per-class macro {s['macro']['mIoU']:.3f}; single-box "
                             f"{one['mIoU']:.3f}; no box {al['no_box']:.1%}; Hit@0.5 {al['Hit@0.5']:.3f}"
                             + (f"; regression k x {s['regression']['kx']:.2f} y {s['regression']['ky']:.2f}"
                                if 'regression' in s else ""))))
        notes.append(f"{name}: paper mAP {pmap} not reproduced (definition not published, model gives no scores).")

    s = _load(os.path.join(d, "findings", "summary_metrics.json"))
    g = _load(os.path.join(d, "findings", "summary_gen.json"))
    ref = {"1/RadCliQ-v1": 1.023, "BLEU": 0.215, "BertScore": 0.441, "SembScore": 0.456,
           "RadGraph": 0.234, "RaTEScore": 0.519, "GREEN": 0.314}
    for m, r in ref.items():
        v = s and s["mapped"].get(m)
        rows.append(("Findings generation", "MIMIC-CXR test (ReXRank)", m, v, r, None,
                     s and s["n"], 2347,
                     (g and m == "1/RadCliQ-v1" and f"extracted as {g['extracted_as']}") or ""))
    notes.append("Findings reference = ReXRank leaderboard row 'CheXOne-R1' (paper main text has no MIMIC "
                 "findings numbers). Metrics computed locally (findings_metrics.py), ReXRank uses CXR-Report-Metric: "
                 "BertScore/BLEU definitions may differ, see findings_metrics.py.")

    lines = ["| Task | Data | Metric | Ours | Paper | n (ours / paper) | vs paper CI | Notes |",
             "|---|---|---|---|---|---|---|---|"]
    print("=" * 140)
    print(f"{'task':<22s}{'metric':<18s}{'ours':>8s}{'paper':>8s}{'n ours/paper':>16s}   {'':<16s}notes")
    print("-" * 140)
    for task, data, metric, v, p, ci, n, pn, note in rows:
        ns = f"{n if n is not None else '—'} / {pn}"
        flag = in_ci(v, ci)
        print(f"{task:<22s}{metric:<18s}{fmt(v, 4 if task.startswith('View') else 3):>8s}"
              f"{fmt(p, 4 if task.startswith('View') else 3):>8s}{ns:>16s}   {flag:<16s}{note or ''}")
        lines.append(f"| {task} | {data} | {metric} | {fmt(v)} | {fmt(p)} | {ns} | {flag} | {note or ''} |")
    print("=" * 140)
    for n_ in notes:
        print("- " + n_)
    lines += [""] + ["- " + n_ for n_ in notes]
    with open(os.path.join(d, "paper_suite_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "paper_suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump([dict(zip(["task", "data", "metric", "ours", "paper", "paper_ci", "n", "paper_n", "note"], r))
                   for r in rows], f, indent=2)
    print(f"-> {d}/paper_suite_summary.md, .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
