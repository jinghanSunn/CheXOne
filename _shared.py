"""
Bootstrap: reuse track8_detection_sized's single-source-of-truth modules.

本目录【不复制】 categories.py / evaluate.py / dicom_utils.py / progress.py /
effusion_infer.py —— 全部从 ../track8_detection_sized 导入, 保证 Qwen 链路与
Faster R-CNN 链路用的是**同一份类别表、同一份官方评分规则、同一份 DICOM 读取**,
指标才可直接对拍。

覆盖路径: 环境变量 TRACK8_SIZED_DIR=/path/to/track8_detection_sized
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SIZED = os.path.normpath(os.path.join(_HERE, "..", "track8_detection_sized"))
SIZED_DIR = os.environ.get("TRACK8_SIZED_DIR", _DEFAULT_SIZED)

if not os.path.isdir(SIZED_DIR):
    raise RuntimeError(
        f"找不到 track8_detection_sized 目录: {SIZED_DIR}\n"
        f"请把本目录放在 code_new/ 下与 track8_detection_sized 同级, "
        f"或设置环境变量 TRACK8_SIZED_DIR 指向它。"
    )

if SIZED_DIR not in sys.path:
    sys.path.insert(0, SIZED_DIR)

# 预导入一次, 启动即暴露路径问题(而不是等到推理到一半才炸)
import categories as _cat          # noqa: E402,F401
import dicom_utils as _dcm         # noqa: E402,F401

NUM_CLASSES = _cat.NUM_CLASSES


def sized_dir():
    return SIZED_DIR
