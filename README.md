# track8_chexone

StanfordAIMI/CheXOne（Qwen2.5-VL-3B 后训练，胸片专用）的评测代码。
从 `track8_detection_qwen/` 拆出来，自包含；唯一的外部依赖是同级的 `../track8_detection_sized`
（类别表、官方评分 `evaluate.py`、DICOM 读取，经 `_shared.py` 引入），所以与 Faster R-CNN、Qwen3.5 的指标仍可直接对拍。

## 三套评测，三个入口

| 入口 | 评测什么 | 对标 |
|---|---|---|
| `run_chexone_eval.sh` | Track 8 的 17 类胸片检测（官方 S_clin） | FRCNN / Qwen3.5 |
| `run_vindr_eval.sh` | VinDr-CXR 官方 test 上的异常定位，逐「图像 + 类别」一个框 | 论文 mIoU 0.452 |
| `run_cxrlt_eval.sh` | CXR-LT 长尾病二分类（5 种病 × 150） | 论文 accuracy 0.816 |

每个脚本 `bash <脚本> help` 看子命令；都建议先 `check` / `probe`，再 `n` / `full`。

```bash
bash run_chexone_eval.sh check && bash run_chexone_eval.sh probe && bash run_chexone_eval.sh n 50
bash run_vindr_eval.sh check    && bash run_vindr_eval.sh probe
bash run_cxrlt_eval.sh check    && bash run_cxrlt_eval.sh probe
```

## 文件

**CheXOne 专用**

| 文件 | 作用 |
|---|---|
| `chexone_infer.py` | Track 8 主入口：per-class「yes/no 闸门 + 单框定位」，输出与 FRCNN 同构的 answer json |
| `chexone_official.py` | 逐行照搬官方 Quick Start 的 runner（VinDr / CXR-LT 用它） |
| `vindr_chexone.py` / `vindr_vis.py` | VinDr 定位评测（mIoU、逐类、分档、回归斜率）与可视化 |
| `cxrlt_chexone.py` | CXR-LT 长尾分类评测（样本构造、解析、准确率、置信区间） |
| `answer_geo.py` | 框几何诊断：面积、中心、跨度、偏左/偏上归因，不占 GPU |
| `iou_report.py` | 逐类 AP + GT 视角 IoU 总表 |

**从 Qwen 链路复制来的共用模块**（改这里不会影响 `track8_detection_qwen/`，反之亦然）

`_shared.py` `parse_utils.py`（含 `<|ref|><|box|>` 与 `\boxed{}` 解析）`image_prep.py`（支持短边 512）
`local_runner.py` `qwen_client.py`（模型别名）`prompt_builder.py`（`CLASS_HINT`）
`compare_answers.py` `visualize_detections.py`

**产物目录**：`answer_chexone/` `cache_chexone/` `png_chexone/` `vis_chexone/` `vis_vindr/` `logs/chexone/`

## 两个口径是训练时定死的，没有开关

1. 坐标归一化到 **0-100**，不是 Qwen 的 0-1000。
2. 输入图先缩到 **短边 512**，processor 不再二次缩放。

## 部署

把整个 `track8_chexone/` 放在服务器 `code_new/` 下，与 `track8_detection_sized/` 同级。
对拍 Qwen 时默认找 `../track8_detection_qwen/answer_qwen`，找不到会自动跳过；也可 `QWEN_ANSWER=/abs/path` 指定。

## 已知限制

- Windows 本机只验证了语法、参数链路、解析和合成数据；**真实模型推理都在服务器上跑**。
- 论文的 VinDr 743 样本、CXR-LT 750 样本没有公开，两个评测的样本构造是近似复现（见各脚本文件头）。

## 论文全链路复现（run_paper_suite.sh）

统一设置：官方 Quick Start runner、reasoning 模式、`max_pixels=512*512`、`max_new_tokens=2048`。

| part | 脚本 | 数据 | 论文 |
|---|---|---|---|
| view | view_chexone.py | MIMIC-CXR test，AP/PA/lateral 各 300 | acc 0.9978 |
| longtail | cxrlt_chexone.py | CXR-LT（MIMIC-CXR-LT）test，5×150 | acc 0.816 |
| phrase | ms_cxr_chexone.py | MS-CXR test | mIoU 0.608 |
| abnormality | vindr_chexone.py | VinDr-CXR test | mIoU 0.452 |
| findings | findings_chexone.py + findings_metrics.py | data/ReXRank_MIMICCXR_test.json（2347） | ReXRank CheXOne-R1 |

```bash
bash run_paper_suite.sh check      # 数据齐不齐
bash run_paper_suite.sh probe      # 每部分几条原始回答
SMALL=1 bash run_paper_suite.sh run   # 每部分 50，先验证链路
bash run_paper_suite.sh run        # 全量（可只跑部分：run view findings）
pip install radeval && bash run_paper_suite.sh metrics   # findings 文本指标
bash run_paper_suite.sh summary    # 总表 -> paper_suite/paper_suite_summary.md
```
