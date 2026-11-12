"""
DICOM -> 送进 Qwen3.5 的 PNG（落盘, 按文件路径喂模型）。

复用 track8_detection_sized/dicom_utils.load_dicom(含 VOI LUT / MONOCHROME1 翻转 /
0.5-99.5 百分位窗宽), 与 Faster R-CNN 链路**完全同一套预处理**, 保证两边比的是
模型差异而不是预处理差异。

额外做三件事:
  1. 限长边(默认 1344px)。送进去的 W_in/H_in 已知 -> 坐标回映射可控。
     注意 processor 还会按 max_pixels 自己再缩一次, 但 Qwen3.5 输出的是
     0-1000 归一化坐标, 对内部缩放不敏感 —— 这是我们能放心预缩的前提。
  2. **落盘到 --png-dir 并缓存 info json**。local 后端(transformers +
     qwen_vl_utils)吃文件路径; 且 PNG 可复用, 重跑不必重解 3000px 的 DICOM。
  3. 可选 CLAHE 对比度增强(--clahe, 需要 opencv), 胸片低对比区域(肋膈角/肺尖)
     对 VLM 更友好。默认关, 避免引入 FRCNN 链路没有的变量。

依赖: Pillow(必需), 可选 opencv-python。
"""
import io
import os
import json
import numpy as np

from _shared import _dcm

load_dicom = _dcm.load_dicom
accession_from_path = _dcm.accession_from_path
iter_dcm_files = _dcm.iter_dcm_files

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _clahe(img_u8, clip=2.0, grid=8):
    try:
        import cv2
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img_u8)
    except Exception:
        return img_u8


def dicom_to_png(dcm_path, max_side=1344, clahe=False, to_rgb=True, short_side=0):
    """返回 (png_bytes, info)。
    info: W0,H0(原图像素), W_in,H_in(实际送模型), ps_col,ps_row(mm/px), acc

    short_side > 0 时**优先**按短边缩放(保持长宽比), 覆盖 max_side。
    CheXOne 训练时的预处理就是这个: VinDr 的 PNG 先缩到短边 512, 再把标注框
    归一化到 0-100。我们用别的口径就是 off-distribution, 所以必须对齐。
    """
    if not _HAS_PIL:
        raise RuntimeError("需要 Pillow: pip install pillow")
    img_u8, meta = load_dicom(dcm_path)
    H0, W0 = img_u8.shape[:2]
    if clahe:
        img_u8 = _clahe(img_u8)

    im = Image.fromarray(img_u8, mode="L")
    if short_side and min(W0, H0) > short_side:
        s = float(short_side) / min(W0, H0)
        W_in, H_in = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
        im = im.resize((W_in, H_in), Image.BILINEAR)
    elif max_side and max(W0, H0) > max_side:
        s = float(max_side) / max(W0, H0)
        W_in, H_in = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
        im = im.resize((W_in, H_in), Image.BILINEAR)
    else:
        W_in, H_in = W0, H0
    if to_rgb:
        im = im.convert("RGB")

    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    info = {
        "W0": W0, "H0": H0, "W_in": W_in, "H_in": H_in,
        "ps_col": meta["pixel_spacing_col_mm"],
        "ps_row": meta["pixel_spacing_row_mm"],
        "acc": accession_from_path(dcm_path) or meta.get("accession_number"),
    }
    return buf.getvalue(), info


def dicom_to_png_file(dcm_path, png_dir, max_side=1344, clahe=False,
                      overwrite=False, meta_cache=True, short_side=0):
    """DICOM -> 落盘 PNG, 返回 (png_path, info)。

    为什么要落盘而不是在内存里转 base64:
      * local 后端(transformers + qwen_vl_utils.process_vision_info)吃的是**文件路径**,
        与 exp5_cava/llm_runner.py 的调用方式一致;
      * PNG 可复用 —— 重跑/换 prompt 不必重新解 DICOM(胸片 3000px 解码不便宜);
      * 导出的图可以直接拿眼睛看, 排查"模型到底看到了什么"比猜强。

    info 里的 W0/H0/ps_col/ps_row 同时缓存成同名 .json, 这样 PNG 命中缓存时
    连 DICOM 头都不用再读。
    """
    os.makedirs(png_dir, exist_ok=True)
    acc = accession_from_path(dcm_path)
    stem = acc or os.path.splitext(os.path.basename(dcm_path))[0]
    png_path = os.path.join(png_dir, f"{stem}.png")
    meta_path = os.path.join(png_dir, f"{stem}.info.json")

    if (not overwrite) and os.path.exists(png_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return png_path, json.load(f)
        except Exception:
            pass   # 缓存坏了就重做

    png_bytes, info = dicom_to_png(dcm_path, max_side=max_side, clahe=clahe,
                                   short_side=short_side)
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    if meta_cache:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
    return png_path, info


def list_images_from_ann(ann_path, image_root):
    """按 COCO json 的 images 列出存在的 dcm 路径, 返回 [(path, image_dict)]。
    与 inference_sized.py 的找文件逻辑一致(含 .dcm/.DCM 后缀补全)。"""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for im in data.get("images", []):
        fp = im.get("file_path") or im.get("file_name")
        if not fp:
            continue
        cand = os.path.join(image_root, fp)
        if not os.path.exists(cand):
            base, _ = os.path.splitext(fp)
            for ext in (".dcm", ".DCM"):
                c2 = os.path.join(image_root, base + ext)
                if os.path.exists(c2):
                    cand = c2
                    break
        if os.path.exists(cand):
            out.append((cand, im))
    return out
