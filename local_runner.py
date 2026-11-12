"""本机 transformers 直跑 Qwen3.5（不需要 vLLM）。

**改编自 `2026WhenSmallModelSpeak/Code/exp5_cava/llm_runner.py`**，刻意保持同一套
加载/生成口径，这样 Track 8 这条链路和 exp5_cava 的 Qwen3.5 行为一致、结果可互相印证。
服务器上跑不了 vLLM，所以这是**默认后端**。

与 exp5 一致的关键点（都是踩过的坑，不要随手改）:
  1. loader = `AutoModelForImageTextToText`（Qwen3.5 是 image-text-to-text，
     不是 `Qwen3VLForConditionalGeneration`；也没有 Qwen3.5-VL 这个 repo）。
  2. `device_map="auto"` + bfloat16 → 配 `CUDA_VISIBLE_DEVICES=0,1,2,3`
     让 27B 自动跨 4×H20 分片，不需要 TP 框架。
  3. `AutoProcessor(min_pixels=..., max_pixels=...)`，单位是 **token 数 × 28 × 28**
     （exp5 默认 256 / 1280 token）。
  4. **`enable_thinking` 必须走 `processor.tokenizer.apply_chat_template`**，
     不能走 `processor.apply_chat_template` —— 某些 transformers 版本下 processor
     会静默吞掉未知 kwargs，导致 thinking 其实一直开着（exp5 文件头原话）。
  5. 图片以**文件路径**放进 messages，交给 `qwen_vl_utils.process_vision_info` 处理。
  6. 即使 enable_thinking=False，也要兜底剥 `<think>...</think>`。

依赖: torch, transformers>=4.57, accelerate, qwen_vl_utils（缺了会降级为直接传 PIL）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*?$", flags=re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """剥 <think>...</think>（含被 max_new_tokens 截断的未闭合块）。"""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048      # grounding 要吐一串 JSON, 比 exp5 的 512 给得多
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    enable_thinking: bool = False


class QwenLocalRunner:
    """Qwen3.5 本机图文生成。接口与 OpenAICompatClient 对齐: chat(image_path, prompt, system)。"""

    def __init__(
        self,
        model_name: str,
        flash_attn: bool = False,
        min_pixels_tokens: int = 256,
        max_pixels_tokens: int = 1280,
        torch_dtype: str = "bfloat16",
        gen_config: Optional[GenerationConfig] = None,
    ):
        self.model_name = model_name
        self.flash_attn = flash_attn
        self.min_pixels_tokens = min_pixels_tokens
        self.max_pixels_tokens = max_pixels_tokens
        self.torch_dtype = torch_dtype
        self._size_logged = False
        self.cfg = gen_config or GenerationConfig()
        self.loader_tag = None      # 实际用哪个类加载的, 打印出来好排查
        self.last_yesno_mass = None # 最近一次 yesno_score 的 P(yes)+P(no)

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._process_vision_info = None
        self._torch = None

    # ------------------------------------------------------------
    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._torch = torch
        dtype = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16,
                 "float32": torch.float32,
                 "auto": "auto"}[self.torch_dtype]
        model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        if self.flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        min_pixels = self.min_pixels_tokens * 28 * 28
        max_pixels = self.max_pixels_tokens * 28 * 28

        # 换模型时最容易栽的一跤: Qwen3.5 走 AutoModelForImageTextToText,
        # 但 Qwen2.5-VL 系(比如 CheXOne)在部分 transformers 版本里没注册进去,
        # 直接 404 / KeyError。所以按优先级试三个入口, 并把实际用上的打出来。
        model, tag, errs = None, None, []
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, **model_kwargs)
            tag = "AutoModelForImageTextToText"
        except Exception as e:  # noqa: BLE001
            errs.append(f"AutoModelForImageTextToText: {type(e).__name__}: {e}")
        if model is None:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name, **model_kwargs)
                tag = "Qwen2_5_VLForConditionalGeneration"
            except Exception as e:  # noqa: BLE001
                errs.append(f"Qwen2_5_VLForConditionalGeneration: {type(e).__name__}: {e}")
        if model is None:
            raise RuntimeError("加载失败, 依次尝试过:\n  " + "\n  ".join(errs))
        self.model, self.loader_tag = model, tag

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=min_pixels, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        cfg = getattr(model, "config", None)
        arch = getattr(cfg, "architectures", None)
        mtype = getattr(cfg, "model_type", None)
        print(f"[INFO] Loaded {tag}: {self.model_name}")
        print(f"[INFO] architectures={arch} model_type={mtype}")
        print(f"[INFO] pixels budget: min={min_pixels} max={max_pixels} "
              f"({self.min_pixels_tokens}/{self.max_pixels_tokens} vision tokens)")

        try:
            from qwen_vl_utils import process_vision_info as _pvi
            self._process_vision_info = _pvi
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] qwen_vl_utils 不可用({e}); 降级为直接向 processor 传 PIL 图")
            self._process_vision_info = None

        if torch.cuda.is_available():
            print(f"[INFO] Model spans {torch.cuda.device_count()} GPU(s)")

    # ------------------------------------------------------------
    def _apply_template(self, messages):
        """enable_thinking 必须打到 tokenizer 层(见文件头第 4 条)。"""
        applier = self.processor.tokenizer.apply_chat_template
        try:
            return applier(messages, tokenize=False, add_generation_prompt=True,
                           enable_thinking=self.cfg.enable_thinking)
        except TypeError:
            return applier(messages, tokenize=False, add_generation_prompt=True)

    def _prepare(self, image_path, prompt, system=None):
        """messages -> (model_inputs, n_prompt_tokens)。chat / yesno_score 共用。"""
        messages = []
        if system:
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}]})

        text = self._apply_template(messages)

        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
        else:
            from PIL import Image
            image_inputs, video_inputs = [Image.open(image_path).convert("RGB")], None

        inputs = self.processor(text=[text], images=image_inputs,
                                videos=video_inputs, padding=True,
                                return_tensors="pt")
        # 实际尺寸打一次就行 —— 以为送进去是 512x615 不代表真拿到它, smart_resize
        # 还会按 min/max_pixels 再动一次(并 round 到 28 的倍数)。亲眼确认比算的可靠。
        if not self._size_logged:
            self._size_logged = True
            try:
                thw = inputs.get("image_grid_thw")
                if thw is not None:
                    _, gh, gw = thw[0].tolist()
                    print(f"[INFO] 实际送入模型的图像尺寸: {int(gw) * 28} x {int(gh) * 28} "
                          f"(grid {int(gw)}x{int(gh)})")
            except Exception:  # noqa: BLE001
                pass
        return inputs.to(self.model.device)

    def chat(self, image_path, prompt, system=None):
        """image_path: 本地图片路径(我们把 DICOM 先导出成 PNG 再传路径, 与 exp5 一致)。"""
        if self.model is None:
            raise RuntimeError("先调用 .load()")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        inputs = self._prepare(image_path, prompt, system)

        gen_kwargs = {"max_new_tokens": self.cfg.max_new_tokens,
                      "do_sample": self.cfg.do_sample}
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
            gen_kwargs["top_p"] = self.cfg.top_p

        with self._torch.no_grad():
            out_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out_ids[0][inputs["input_ids"].shape[-1]:]
        # thinking 开着时保留特殊 token, 让下游能看到 <think>
        skip_special = not self.cfg.enable_thinking
        out = self.processor.decode(trimmed, skip_special_tokens=skip_special)
        # 兜底: 即便 enable_thinking=False 某些 checkpoint 仍会漏 <think>
        return strip_thinking(out)

    # ------------------------------------------------------------
    _YES_FORMS = ("Yes", "yes", "YES", " Yes", " yes", "Yes.", "yes.")
    _NO_FORMS = ("No", "no", "NO", " No", " no", "No.", "no.")

    def _tok_ids(self, forms):
        """表面形式 -> 首 token id 集合(去重)。多 token 的只取第一个, 够用。"""
        ids = []
        for f in forms:
            enc = self.tokenizer.encode(f, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
        return sorted(set(ids))

    def yesno_score(self, image_path, question, system=None):
        """问一个是非题, 返回 (p_yes, decoded_text)。

        为什么要这个: CheXOne 的 abnormality grounding 只吐坐标、**不带置信度**,
        而 mAP 是按置信度排序算 PR 曲线的 —— 全给同一个分数等于随机排序, AP 会
        严重失真。用一个独立的 yes/no VQA 拿 P(yes) 当分数, 既给了可排序的量,
        也顺带当了"这类到底有没有"的闸门(省掉大量无谓的 localize 请求)。

        实现: 一次 forward, 取最后位置的**原始** logits, 在 {yes} vs {no} 上做 softmax。
        不用 generate(output_scores=True): 那拿到的是过完 logits processor 的分数,
        CheXOne 的 generation_config 带 repetition_penalty=1.05, 而问题里正好有
        " yes"/" no" 两个 token —— 会被惩罚, P(yes) 被系统性扭曲。
        必须走 instruct 模式(不要追加 \boxed 推理指令), 否则第一个 token 是
        推理正文的开头而不是答案。

        self.last_yesno_mass = P(yes)+P(no) 的原始概率质量。很低(<0.5)说明模型
        第一个 token 根本不想答 yes/no, 此时 P(yes) 只是噪声, 不该当闸门用。
        """
        if self.model is None:
            raise RuntimeError("先调用 .load()")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        inputs = self._prepare(image_path, question, system)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        probs = self._torch.softmax(logits, dim=-1)
        yes_ids, no_ids = self._tok_ids(self._YES_FORMS), self._tok_ids(self._NO_FORMS)
        p_yes = float(sum(probs[i].item() for i in yes_ids))
        p_no = float(sum(probs[i].item() for i in no_ids))
        denom = p_yes + p_no
        self.last_yesno_mass = denom
        p = (p_yes / denom) if denom > 1e-9 else 0.5
        txt = self.tokenizer.decode([int(self._torch.argmax(logits).item())],
                                    skip_special_tokens=True)
        return p, txt.strip()
