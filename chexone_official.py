"""CheXOne inference, copied from the official Quick Start (README / HF model card).

Nothing here is ours except the class wrapper and the pixel setting. Every call below is the
call in the official snippet, in the same order:

  Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype="auto", device_map="auto")
  AutoProcessor.from_pretrained(path[, min_pixels=..., max_pixels=...])
  processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  process_vision_info(messages)
  processor(text=[text], images=..., videos=..., padding=True, return_tensors="pt")
  inputs.to("cuda")
  model.generate(**inputs, max_new_tokens=1024)        # no do_sample / temperature: the model's
                                                       # own generation_config applies
  processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

How this differs from local_runner.QwenLocalRunner (the one used for Qwen3.5):
  loader        AutoModelForImageTextToText first      -> Qwen2_5_VLForConditionalGeneration
  dtype         bfloat16                               -> "auto" (what the checkpoint was saved in)
  template      tokenizer.apply_chat_template(+enable_thinking, a Qwen3.5 kwarg) -> processor.apply_chat_template
  generation    do_sample=False, 256 tokens (instruct) -> generation_config.json as shipped
                (do_sample=true, temperature=1e-6, repetition_penalty=1.05), 1024 tokens
  decode        processor.decode + <think> stripping   -> batch_decode(clean_up_tokenization_spaces=False)
  pixel budget  max 640 tokens = 501,760 px            -> max_pixels=512*512 = 262,144 px (README "rec")

The README says the processor default is 4-16384 visual tokens and recommends
max_pixels=512*512 "to align with the training setting"; the paper's evaluation script
(examples/train/chexone/prepare_testing/3_inference_code/CheXOne.sh) also sets MAX_PIXELS=262144.
So pixels="rec" is the default and pixels="default" is the bare processor.

The paper's script runs through ms-swift + vLLM with max_new_tokens=2048, temperature=0 and the
system prompt 'You are a helpful assistant.' (the Qwen2.5-VL chat template adds that one itself).
"""
import os


class OfficialRunner:
    """chat(image_path, prompt) -> str, same interface as QwenLocalRunner."""

    MIN_PIXELS = 256 * 28 * 28      # README comment
    MAX_PIXELS = 512 * 512          # README comment, paper script MAX_PIXELS=262144

    def __init__(self, model_path, pixels="rec", max_new_tokens=1024, flash_attn=False):
        self.model_path = model_path
        self.pixels = pixels
        self.max_new_tokens = max_new_tokens
        self.flash_attn = flash_attn
        self.model = self.processor = self._pvi = None
        self._size_logged = False

    def load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        if self.flash_attn:     # the README's commented alternative
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map="auto")
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto")
        if self.pixels == "rec":
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, min_pixels=self.MIN_PIXELS, max_pixels=self.MAX_PIXELS)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._pvi = process_vision_info
        gc = self.model.generation_config
        print(f"[official] loaded {self.model_path}  pixels={self.pixels}  "
              f"max_new_tokens={self.max_new_tokens}")
        print(f"[official] generation_config: do_sample={gc.do_sample} "
              f"temperature={gc.temperature} repetition_penalty={gc.repetition_penalty}")

    def chat(self, image_path, prompt):
        """image_path: one path, or a list of paths (one study, several views; same message format
        as the official snippet, just more {"type": "image"} entries before the text)."""
        paths = list(image_path) if isinstance(image_path, (list, tuple)) else [image_path]
        for p in paths:
            if not p or not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")
        messages = [{"role": "user", "content":
                     [{"type": "image", "image": p} for p in paths]
                     + [{"type": "text", "text": prompt}]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._pvi(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")

        if not self._size_logged:       # logging only, does not touch inference
            self._size_logged = True
            try:
                _, gh, gw = inputs["image_grid_thw"][0].tolist()
                print(f"[official] effective input {int(gw) * 28}x{int(gh) * 28} px "
                      f"({int(gh) * int(gw)} patches -> {int(gh) * int(gw) // 4} visual tokens)")
            except Exception:  # noqa: BLE001
                pass

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        return output_text[0]
