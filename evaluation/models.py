"""
Centralized Model Loading Utilities.

Provides a unified interface for loading models across all experiments
and runners, handling device placement, authentication, and caching.
"""

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
    """
    Load a model and tokenizer with unified configuration.

    Args:
        model_name: Model identifier (key from MODELS or HuggingFace ID)
        device: Target device ("auto", "cuda", "cpu", or specific like "cuda:0")
        dtype: Model dtype (None for auto-selection based on device)
        hf_token: HuggingFace token for gated models (falls back to HF_TOKEN env var)
        trust_remote_code: Whether to trust remote code (required for some models)

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model("gpt2")
        >>> model, tokenizer = load_model("llama-3.1-8b", device="cuda")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve model name to HuggingFace ID
    if model_name in MODELS:
        hf_id = MODELS[model_name]["hf_id"]
        requires_auth = MODELS[model_name].get("requires_auth", False)
    else:
        # Assume it's a direct HuggingFace ID
        hf_id = model_name
        requires_auth = "llama" in model_name.lower() or "meta-llama" in model_name.lower()

    # Get HF token
    token = hf_token or os.environ.get("HF_TOKEN")
    if requires_auth and not token:
        raise ValueError(
            f"Model {model_name} requires authentication. "
            "Set HF_TOKEN environment variable or pass hf_token parameter."
        )

    # Resolve device
    resolved_device = _resolve_device(device)

    # Resolve dtype
    resolved_dtype = _resolve_dtype(dtype, resolved_device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "token": token,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolved_dtype,
    }

    # Use device_map for large models on CUDA
    if resolved_device != "cpu" and _is_large_model(model_name):
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    # Move to device if not using device_map
    if model_kwargs["device_map"] is None and resolved_device != "cpu":
        model = model.to(resolved_device)

    model.eval()

    return model, tokenizer


def _resolve_device(device: str) -> str:
    """Resolve device string to actual device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(
    dtype: Optional[torch.dtype],
    device: str,
) -> torch.dtype:
    """Resolve dtype based on device if not specified."""
    if dtype is not None:
        return dtype

    # Use FP16 on CUDA, FP32 on CPU
    if device == "cpu":
        return torch.float32
    return torch.float16


def _is_large_model(model_name: str) -> bool:
    """Check if model is large enough to require device_map."""
    if model_name in MODELS:
        # Models with >1B parameters
        return MODELS[model_name].get("layers", 0) > 20
    # Heuristic for unknown models
    return any(x in model_name.lower() for x in ["7b", "8b", "13b", "70b", "llama"])


def get_model_info(model_name: str) -> dict:
    """Get model metadata from registry."""
    if model_name in MODELS:
        return MODELS[model_name].copy()
    return {"hf_id": model_name, "type": "unknown"}


def get_device_for_model(model_name: str) -> str:
    """Get recommended GPU type for a model."""
    if model_name in MODELS:
        return MODELS[model_name].get("gpu", "T4")
    return "T4"
