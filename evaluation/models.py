import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .constants import MODELS, DEFAULT_MODEL


def _check_flash_attention_available():
    """Check if flash-attn is installed and CUDA is available."""
    try:
        import flash_attn
        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model(
    model_name=DEFAULT_MODEL,
    device="auto",
    dtype=None,
    hf_token=None,
    trust_remote_code=True,
    use_flash_attention=True,
    use_torch_compile=False,
    compile_mode="reduce-overhead",
):
    """
    Load a model with optional performance optimizations.

    Args:
        model_name: Model identifier (from MODELS dict or HuggingFace ID)
        device: Device to load model on ("auto", "cuda", "cpu")
        dtype: Model dtype (None for auto-detect)
        hf_token: HuggingFace token for gated models
        trust_remote_code: Allow custom model code
        use_flash_attention: Enable Flash Attention 2 if available (~1.5x speedup)
        use_torch_compile: Enable torch.compile (~1.3x speedup, longer warmup)
        compile_mode: torch.compile mode ("reduce-overhead", "max-autotune", "default")

    Returns:
        model, tokenizer tuple
    """
    if model_name in MODELS:
        hf_id = MODELS[model_name]["hf_id"]
        requires_auth = MODELS[model_name].get("requires_auth", False)
    else:
        hf_id = model_name
        requires_auth = (
            "llama" in model_name.lower() or "meta-llama" in model_name.lower()
        )

    token = hf_token or os.environ.get("HF_TOKEN")
    if requires_auth and not token:
        raise ValueError(
            f"Model {model_name} requires authentication. "
            "Set HF_TOKEN environment variable or pass hf_token parameter."
        )

    resolved_device = _resolve_device(device)

    resolved_dtype = _resolve_dtype(dtype, resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "token": token,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolved_dtype,
    }

    # Enable Flash Attention 2 if requested and available
    if use_flash_attention and _check_flash_attention_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if _is_large_model(model_name):
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    if model_kwargs["device_map"] is None:
        model = model.to(resolved_device)

    model.eval()

    # Apply torch.compile for faster repeated inference
    if use_torch_compile:
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), using eager mode")

    return model, tokenizer


def _resolve_device(device):
    if device == "auto":
        return "cuda"
    return device


def _resolve_dtype(dtype, device):
    if dtype is not None:
        return dtype

    return torch.float16


def _is_large_model(model_name):
    if model_name in MODELS:
        return MODELS[model_name].get("layers", 0) > 20

    return any(x in model_name.lower() for x in ["7b", "8b", "13b", "70b", "llama"])


def get_model_info(model_name):
    if model_name in MODELS:
        return MODELS[model_name].copy()
    return {"hf_id": model_name, "type": "unknown"}


def get_device_for_model(model_name):
    if model_name in MODELS:
        return MODELS[model_name].get("gpu", "T4")
    return "T4"
