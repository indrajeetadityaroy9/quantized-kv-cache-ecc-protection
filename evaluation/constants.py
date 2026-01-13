"""
Experiment Configuration Constants for ECC-Protected KV Cache Evaluation.

This module defines the canonical configuration for BER sweep experiments,
including cache mode definitions, error rate levels, and model specifications.

Cache Modes:
    Each cache mode specifies:
    - bits: Storage bits per value (for bandwidth calculations)
    - description: Human-readable description for plots/tables
    - protected: Whether ECC protection is applied (for grouping)

BER Levels:
    Standard: [0, 1e-4, 1e-3, 1e-2]
    Extended: [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    BER=0 serves as corruption-free baseline for each codec.

Seeds:
    10 seeds for NeurIPS/ICML statistical rigor:
    [42, 101, 997, 1999, 4999, 7919, 10007, 15073, 21001, 31337]

    These are prime numbers to minimize correlation artifacts.

Models:
    Configurations for GPT-2 (T4), Mistral-7B (A100-40GB), LLaMA-3.1-8B (A100-80GB).

GPU Bandwidth:
    Peak memory bandwidth values for bandwidth efficiency calculations.
    Used to contextualize throughput measurements.

Usage:
    from evaluation.constants import get_cache_modes, get_ber_levels, get_seeds

    for mode in get_cache_modes(protected_only=True):
        for ber in get_ber_levels():
            for seed in get_seeds():
                run_experiment(mode, ber, seed)
"""
CACHE_MODES = {
    "fp16": {
        "bits": 16,
        "description": "FP16 Oracle baseline (no quantization)",
        "protected": False,
    },
    "fp8": {
        "bits": 8,
        "description": "FP8 E4M3 quantization (vLLM standard)",
        "protected": False,
    },
    "int4": {
        "bits": 4,
        "description": "INT4 Unprotected (no error correction)",
        "protected": False,
    },
    "int4-hamming": {
        "bits": 7,
        "description": "INT4 + Hamming(7,4) SEC",
        "protected": True,
    },
    "int4-hamming84": {
        "bits": 8,
        "description": "INT4 + Hamming(8,4) SECDED (keeps corrupted)",
        "protected": True,
    },
    "int4-hamming84-interp": {
        "bits": 8,
        "description": "INT4 + Hamming(8,4) SECDED + Linear Interpolation",
        "protected": True,
    },
    "int12-golay": {
        "bits": 8,
        "description": "INT4 triplets + Golay(24,12) - corrects up to 3 errors",
        "protected": True,
    },
}


CACHE_MODE_ORDER = [
    "fp16",
    "fp8",
    "int4",
    "int4-hamming",
    "int4-hamming84",
    "int4-hamming84-interp",
    "int12-golay",
]


BER_LEVELS = [0, 1e-4, 1e-3, 1e-2]


BER_LEVELS_EXTENDED = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


DEFAULT_CONFIG = {
    "max_length": 256,
    "stride": 128,
    "block_size": 32,
    "max_samples": 50,
    # 10 seeds for statistical rigor (NeurIPS/ICML standard)
    "seeds": [42, 101, 997, 1999, 4999, 7919, 10007, 15073, 21001, 31337],
}


MODELS = {
    "gpt2": {
        "hf_id": "gpt2",
        "type": "gpt2",
        "layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_kv_heads": 12,  # No GQA
        "requires_auth": False,
        "gpu": "T4",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "type": "mistral",
        "layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,  # GQA: 4 query heads per KV head
        "requires_auth": False,
        "gpu": "A100-40GB",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "type": "llama",
        "layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,  # GQA: 4 query heads per KV head
        "requires_auth": True,
        "gpu": "A100-80GB",
    },
}


DEFAULT_MODEL = "gpt2"


CACHE_MODE_LABELS = {
    "fp16": "FP16 (Oracle)",
    "fp8": "FP8 (E4M3)",
    "int4": "INT4 (Unprotected)",
    "int4-hamming": "Hamming(7,4)",
    "int4-hamming84": "Hamming(8,4)",
    "int4-hamming84-interp": "H(8,4)+Interp",
    "int12-golay": "Golay(24,12)",
}


# Canonical mode configuration for ECC shim
# Maps cache_mode -> ECCShimConfig parameters
# This is the single source of truth for mode configurations
MODE_CONFIG = {
    "fp16": {"codec": "fp16", "use_interpolation": False},
    "fp8": {"codec": "fp8", "use_interpolation": False},
    "int4": {"codec": "int4", "use_interpolation": False},
    "int4-hamming": {"codec": "hamming74", "use_interpolation": False},
    "int4-hamming84": {"codec": "hamming84", "use_interpolation": False},
    "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True},
    "int12-golay": {"codec": "golay", "use_interpolation": False},
}


def get_mode_config(cache_mode: str) -> dict:
    """Get ECC shim configuration for a cache mode.

    Args:
        cache_mode: One of the valid cache modes (fp16, fp8, int4, int4-hamming, etc.)

    Returns:
        Dict with 'codec' and 'use_interpolation' keys for ECCShimConfig.

    Raises:
        ValueError: If cache_mode is not recognized.
    """
    if cache_mode not in MODE_CONFIG:
        raise ValueError(
            f"Unknown cache mode: {cache_mode}. Valid modes: {list(MODE_CONFIG.keys())}"
        )
    return MODE_CONFIG[cache_mode].copy()


GPU_BANDWIDTH_GBPS = {
    "T4": 320.0,
    "V100": 900.0,
    "A100-40GB": 1555.0,
    "A100-80GB": 2039.0,
    "A10G": 600.0,
    "L4": 300.0,
    "H100": 3352.0,
    "RTX3090": 936.2,
    "RTX4090": 1008.0,
}


DEFAULT_GPU = "A100-80GB"


def get_gpu_bandwidth(gpu_type=None):
    if gpu_type is None:
        gpu_type = DEFAULT_GPU
    return GPU_BANDWIDTH_GBPS.get(gpu_type, GPU_BANDWIDTH_GBPS[DEFAULT_GPU])


def compute_bandwidth_efficiency(
    throughput_mvalues_sec,
    bytes_per_value=1,
    gpu_type=None,
):
    peak_bandwidth = get_gpu_bandwidth(gpu_type)

    achieved_bandwidth = throughput_mvalues_sec * bytes_per_value / 1000.0
    return 100.0 * achieved_bandwidth / peak_bandwidth


def get_cache_modes(protected_only=False):
    if protected_only:
        return [m for m in CACHE_MODE_ORDER if CACHE_MODES[m]["protected"]]
    return CACHE_MODE_ORDER.copy()


def get_ber_levels(extended=False):
    if extended:
        return BER_LEVELS_EXTENDED.copy()
    return BER_LEVELS.copy()


def get_seeds():
    return DEFAULT_CONFIG["seeds"].copy()
