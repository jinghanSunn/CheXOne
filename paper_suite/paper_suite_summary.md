| Task | Data | Metric | Ours | Paper | n (ours / paper) | vs paper CI | Notes |
|---|---|---|---|---|---|---|---|
| View classification | MIMIC-CXR test | Accuracy | 0.988 | 0.998 | 900 / 900 | within paper CI | unparsed 0.0%; recall {'AP': 0.9833333333333333, 'PA': 0.98, 'Lateral': 1.0} |
| Long-tail disease | CXR-LT (MIMIC) test | Accuracy | 0.757 | 0.816 | 750 / 750 | below paper CI | sens 0.805 spec 0.709 pred_yes 0.55; 75 pos + 75 neg per disease (paper ratio unknown) |
| Phrase grounding | MS-CXR test | mIoU (per query) | 0.427 | 0.608 | 176 / 127 | below paper CI | per-class macro 0.378; single-box 0.448; no box 0.0%; Hit@0.5 0.347; regression k x 0.92 y 0.78 |
| Abnormality grounding | VinDr-CXR test | mIoU (per query) | 0.250 | 0.452 | 2108 / 743 | below paper CI | per-class macro 0.200; single-box 0.257; no box 0.0%; Hit@0.5 0.188; regression k x 0.61 y 0.66 |
| Findings generation | MIMIC-CXR test (ReXRank) | 1/RadCliQ-v1 | — | 1.023 | 2347 / 2347 |  | extracted as {'boxed_unclosed': 17, 'boxed': 2330} |
| Findings generation | MIMIC-CXR test (ReXRank) | BLEU | 0.204 | 0.215 | 2347 / 2347 |  |  |
| Findings generation | MIMIC-CXR test (ReXRank) | BertScore | 0.576 | 0.441 | 2347 / 2347 |  |  |
| Findings generation | MIMIC-CXR test (ReXRank) | SembScore | — | 0.456 | 2347 / 2347 |  |  |
| Findings generation | MIMIC-CXR test (ReXRank) | RadGraph | — | 0.234 | 2347 / 2347 |  |  |
| Findings generation | MIMIC-CXR test (ReXRank) | RaTEScore | — | 0.519 | 2347 / 2347 |  |  |
| Findings generation | MIMIC-CXR test (ReXRank) | GREEN | — | 0.314 | 2347 / 2347 |  |  |

- Phrase grounding: paper mAP 0.835 not reproduced (definition not published, model gives no scores).
- Abnormality grounding: paper mAP 0.471 not reproduced (definition not published, model gives no scores).
- Findings reference = ReXRank leaderboard row 'CheXOne-R1' (paper main text has no MIMIC findings numbers). Metrics computed locally (findings_metrics.py), ReXRank uses CXR-Report-Metric: BertScore/BLEU definitions may differ, see findings_metrics.py.
