"""
Qwen3.5 推理后端封装 —— 统一成一个 chat(image_path, prompt, system) -> text。

  backend=local     【默认】本机 transformers 直跑, 不需要 vLLM。
                    CUDA_VISIBLE_DEVICES=0,1,2,3 让 device_map="auto" 把 27B
                    自动跨 4×H20 分片。加载/生成口径与
                    2026WhenSmallModelSpeak/Code/exp5_cava/llm_runner.py 一致。
                    权重路径见下方 DEFAULT_MODEL(唯一事实源)。
  backend=dashscope 阿里云百炼 OpenAI 兼容模式(不占本地 GPU, 适合先验证 prompt)
                    --model qwen3.5-plus, key 取 $DASHSCOPE_API_KEY
  backend=openai    任意 OpenAI 兼容端点(--base-url 自己给)
  backend=vllm      OpenAI 兼容端点的别名, 默认 http://127.0.0.1:8000/v1
                    (服务器上 vLLM 用不了, 留着给以后/别的机器)

⚠️ 不存在 "Qwen3.5-VL" 这个 repo。3.5 这代把视觉并进主干(early fusion),
   模型名就是 Qwen3.5-9B / -27B / -35B-A3B / -122B-A10B / -397B-A17B,
   pipeline_tag = image-text-to-text。写成 Qwen3.5-VL-xx 会 404。

图片一律按**本地文件路径**传(与 exp5_cava 一致); HTTP 后端内部再读文件转 base64。
HTTP 走 stdlib urllib(零依赖), 带指数退避重试。
"""
import os
import json
import time
import base64
import urllib.request
import urllib.error


class QwenClientError(RuntimeError):
    pass


# ==========================================================================
# 【模型配置唯一事实源】—— run_qwen_eval.sh 反查这里, 不另写一份。
#
#   MODEL_ROOT   权重所在目录(与 exp5_cava 用的是同一个根)
#   DEFAULT_SIZE 默认跑哪个尺寸, 换模型改这一个词就够了
#   DEFAULT_MODEL 由上面两个派生; 也可以整条路径直接覆盖
#
# 环境变量覆盖(优先级: QWEN_MODEL > QWEN_SIZE > 下面的字面量):
#   export QWEN_MODEL_ROOT=/mnt/data/jinghansun/Huggingface/Qwen
#   export QWEN_SIZE=9B                     # 调试期用小模型
#   export QWEN_MODEL=/完全自定义/的/路径     # 不在 ROOT 下时用这个
# ==========================================================================
MODEL_ROOT = (os.environ.get("QWEN_MODEL_ROOT")
              or "/mnt/data/jinghansun/Huggingface/Qwen").rstrip("/\\")

DEFAULT_SIZE = os.environ.get("QWEN_SIZE") or "27B"

# 尺寸简写 -> 目录名。调试期用 9B, 正式跑换 27B, 不用每次贴长路径:
#     MODEL=9B bash run_qwen_eval.sh smoke
#     python qwen_infer.py --model 9B
SIZE_ALIASES = {
    "0.8b": "Qwen3.5-0.8B", "2b": "Qwen3.5-2B", "4b": "Qwen3.5-4B",
    "9b": "Qwen3.5-9B", "27b": "Qwen3.5-27B",
    "35b": "Qwen3.5-35B-A3B", "35b-a3b": "Qwen3.5-35B-A3B",
    "122b": "Qwen3.5-122B-A10B", "122b-a10b": "Qwen3.5-122B-A10B",
    "397b": "Qwen3.5-397B-A17B", "397b-a17b": "Qwen3.5-397B-A17B",
}


def _under_root(name):
    # 固定用正斜杠拼: 目标是 Linux 服务器, 而在 Windows 上调试时 os.path.join
    # 会塞进反斜杠, 让路径变成 /mnt/.../Qwen\Qwen3.5-9B 这种混合货。
    return MODEL_ROOT + "/" + name.strip("/\\")


# 非 Qwen3.5 系的模型(目前只有 CheXOne)放这里, 值可以是 HF repo id 或本地路径。
# 优先级: 环境变量 > 字面量。
#   export CHEXONE_MODEL=/mnt/data/jinghansun/Huggingface/CheXOne
MODEL_ALIASES = {
    "chexone": os.environ.get("CHEXONE_MODEL") or "StanfordAIMI/CheXOne",
}


def resolve_model(name):
    """把 '9B' / 'Qwen3.5-9B' 这种简写展开成 MODEL_ROOT 下的全路径。
    已经是路径(含分隔符)或 API 模型名(含 '-plus'/'-flash')就原样返回。"""
    if not name:
        return name
    if os.sep in name or "/" in name:      # 已经是路径
        return name
    low = name.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    if low in SIZE_ALIASES:
        return _under_root(SIZE_ALIASES[low])
    if low.startswith("qwen3.5-") and not low.endswith(("-plus", "-flash")):
        return _under_root(name)
    return name                            # dashscope 的 qwen3.5-plus 等, 原样


DEFAULT_MODEL = os.environ.get("QWEN_MODEL") or resolve_model(DEFAULT_SIZE)


def list_local_models():
    """列出 MODEL_ROOT 下实际存在的 Qwen3.5-* 权重目录(判据: 有 config.json)。"""
    out = []
    try:
        for d in sorted(os.listdir(MODEL_ROOT)):
            p = _under_root(d)
            if os.path.isdir(p) and (d.lower().startswith("qwen")
                                     or d.lower().startswith("chexone")):
                ok = os.path.exists(p.rstrip("/") + "/config.json")
                out.append((d, p, ok))
    except Exception:
        pass
    return out


def _file_to_data_url(path):
    ext = os.path.splitext(path)[1].lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


# --------------------------------------------------------------------------
# OpenAI 兼容后端 (dashscope / 任意网关 / vllm)
# --------------------------------------------------------------------------
class OpenAICompatClient:
    def __init__(self, base_url, model, api_key=None, temperature=0.0,
                 max_tokens=2048, timeout=180, retries=3, enable_thinking=False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.enable_thinking = enable_thinking

    def _post(self, payload):
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat(self, image_path, prompt, system=None):
        content = [
            {"type": "image_url", "image_url": {"url": _file_to_data_url(image_path)}},
            {"type": "text", "text": prompt},
        ]
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
        }
        last = None
        for i in range(self.retries):
            try:
                resp = self._post(payload)
                return resp["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "ignore")[:400]
                except Exception:
                    pass
                last = f"HTTP {e.code}: {body}"
                # 有些网关不认 chat_template_kwargs -> 去掉重试
                if e.code == 400 and "chat_template_kwargs" in payload:
                    payload.pop("chat_template_kwargs", None)
                    continue
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * (2 ** i))
        raise QwenClientError(f"chat failed after {self.retries} tries: {last}")


# --------------------------------------------------------------------------
def build_client(args):
    """根据 argparse 的 args 造客户端(已 load 好, 可直接 .chat())。"""
    args.model = resolve_model(args.model)      # '9B' -> 全路径
    if args.backend == "local":
        from local_runner import QwenLocalRunner, GenerationConfig
        runner = QwenLocalRunner(
            model_name=args.model,
            flash_attn=args.flash_attn,
            min_pixels_tokens=args.min_pixels_tokens,
            max_pixels_tokens=args.max_pixels_tokens,
            torch_dtype=args.dtype,
            gen_config=GenerationConfig(
                max_new_tokens=args.max_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                enable_thinking=args.thinking),
        )
        runner.load()
        return runner

    if args.backend in ("vllm", "openai", "dashscope"):
        base = args.base_url
        if args.backend == "dashscope" and not base:
            base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if args.backend == "vllm" and not base:
            base = "http://127.0.0.1:8000/v1"
        key = args.api_key or os.environ.get(
            "DASHSCOPE_API_KEY" if args.backend == "dashscope" else "OPENAI_API_KEY")
        return OpenAICompatClient(base, args.model, key,
                                  temperature=args.temperature,
                                  max_tokens=args.max_tokens,
                                  timeout=args.timeout,
                                  retries=args.retries,
                                  enable_thinking=args.thinking)
    raise ValueError(f"unknown backend: {args.backend}")


def add_client_args(ap):
    ap.add_argument("--backend", default="local",
                    choices=["local", "dashscope", "openai", "vllm"],
                    help="local=本机transformers直跑(默认, 服务器用不了vllm); "
                         "dashscope=阿里云百炼; openai/vllm=OpenAI兼容端点")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"本地权重路径, 或尺寸简写 9B/27B/35B-A3B/122B-A10B "
                         f"(会在 {MODEL_ROOT} 下展开); dashscope 下用 qwen3.5-plus。"
                         f"默认 {DEFAULT_MODEL} (改 qwen_client.DEFAULT_MODEL "
                         f"或 export QWEN_MODEL=...)。没有 Qwen3.5-VL 这个名字")
    ap.add_argument("--base-url", default=None, help="仅 openai/vllm/dashscope 后端")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="grounding 要吐一串 JSON, 别给太小; 开 --thinking 时建议 >=4096")
    ap.add_argument("--timeout", type=int, default=180, help="仅 HTTP 后端")
    ap.add_argument("--retries", type=int, default=3, help="仅 HTTP 后端")
    ap.add_argument("--thinking", action="store_true",
                    help="开 Qwen3.5 思考模式(更准但慢且费token; 解析层会剥 <think>)")
    # ---- 仅 local 后端 ----
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32", "auto"],
                    help="auto=跟 config.json 走(CheXOne 官方示例用的就是 auto)")
    ap.add_argument("--flash-attn", action="store_true",
                    help="用 flash_attention_2(需装 flash-attn)")
    ap.add_argument("--min-pixels-tokens", type=int, default=256,
                    help="processor min_pixels = N*28*28, 与 exp5_cava 默认一致")
    ap.add_argument("--max-pixels-tokens", type=int, default=1280,
                    help="processor max_pixels = N*28*28。胸片细节多, 调到 2048-4096 "
                         "能提高小病灶可见度, 代价是显存/时延")
    return ap


if __name__ == "__main__":
    # 列出本机可用的 Qwen 权重, 方便挑调试用的小模型
    print(f"MODEL_ROOT    = {MODEL_ROOT}"
          f"{'   (QWEN_MODEL_ROOT)' if os.environ.get('QWEN_MODEL_ROOT') else ''}")
    print(f"DEFAULT_SIZE  = {DEFAULT_SIZE}"
          f"{'   (QWEN_SIZE)' if os.environ.get('QWEN_SIZE') else ''}")
    print(f"DEFAULT_MODEL = {DEFAULT_MODEL}"
          f"{'   (QWEN_MODEL)' if os.environ.get('QWEN_MODEL') else ''}")
    rows = list_local_models()
    if not rows:
        print(f"\n[!] {MODEL_ROOT} 下没扫到 Qwen* 目录(路径不对或权重没下)")
    else:
        print(f"\n  {'目录':<26s}{'config.json':<14s}路径")
        for d, p, ok in rows:
            print(f"  {d:<26s}{'有' if ok else '缺':<14s}{p}")
    print("\n尺寸简写(--model 9B 即可, 不必贴全路径):")
    print("  " + "  ".join(sorted(set(SIZE_ALIASES.values()))))
