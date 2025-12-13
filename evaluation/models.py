import os
from typing import Optional, Tuple, Union

import torch

from .constants import MODELS, DEFAULT_MODEL


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    if _is_large_model(model_name):
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    if model_kwargs["device_map"] is None:
        model = model.to(resolved_device)

    model.eval()

    return model, tokenizer


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda"
    return device


def _resolve_dtype(
    dtype: Optional[torch.dtype],
    device: str,
) -> torch.dtype:
    if dtype is not None:
        return dtype

    return torch.float16


def _is_large_model(model_name: str) -> bool:
    if model_name in MODELS:
        return MODELS[model_name].get("layers", 0) > 20

    return any(x in model_name.lower() for x in ["7b", "8b", "13b", "70b", "llama"])


def get_model_info(model_name: str) -> dict:
    if model_name in MODELS:
        return MODELS[model_name].copy()
    return {"hf_id": model_name, "type": "unknown"}


def get_device_for_model(model_name: str) -> str:
    if model_name in MODELS:
        return MODELS[model_name].get("gpu", "T4")
    return "T4"
