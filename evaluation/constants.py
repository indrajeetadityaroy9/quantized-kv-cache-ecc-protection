# Cache modes with vLLM kv_cache_dtype mappings
# Note: "int4" (unprotected) and "int4-hamming84-interp" not yet implemented
CACHE_MODES = {
    "fp16": {
        "bits": 16,
        "description": "FP16 Oracle baseline (no quantization)",
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
    "int12-golay": {
        "bits": 8,
        "description": "INT4 triplets + Golay(24,12) - corrects up to 3 errors",
        "protected": True,
    },
    "int4-golay-hybrid": {
        "bits": 8,
        "description": "Hybrid Golay(24,12) triplets + Hamming(8,4) remainder",
        "protected": True,
    },
    "int4-reed-solomon": {
        "bits": 6,  # 48 bits / 8 values = 6 bits/value
        "description": "INT4 + Reed-Solomon(12,8) - corrects up to 2 symbols (8 bits)",
        "protected": True,
    },
}


# Note: "int4" (unprotected) and "int4-hamming84-interp" not yet implemented in vLLM
CACHE_MODE_ORDER = [
    "fp16",
    "int4-hamming",
    "int4-hamming84",
    "int12-golay",
    "int4-golay-hybrid",
    "int4-reed-solomon",
]


# Publication-quality BER levels (extended range for comprehensive analysis)
BER_LEVELS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# Alias for backward compatibility
BER_LEVELS_EXTENDED = BER_LEVELS


# Publication-quality default configuration
# These values are set for statistically significant results
DEFAULT_CONFIG = {
    "max_length": 512,      # Longer sequences for proper perplexity measurement
    "stride": 256,          # Half-overlap for sliding window
    "block_size": 32,
    "max_samples": 100,     # Minimum for statistical significance
    "seeds": [42, 101, 997, 1999, 4999],  # 5 seeds for robust statistics
}

# Minimum thresholds for valid evaluation (enforced at runtime)
MIN_SAMPLES = 50
MIN_SEEDS = 3
MIN_MAX_LENGTH = 256


MODELS = {
    # GPT-2 family
    "gpt2": {
        "hf_id": "gpt2",
        "type": "gpt2",
        "layers": 12,
        "hidden_size": 768,
        "requires_auth": False,
        "gpu": "T4",
    },
    "gpt2-medium": {
        "hf_id": "gpt2-medium",
        "type": "gpt2",
        "layers": 24,
        "hidden_size": 1024,
        "requires_auth": False,
        "gpu": "T4",
    },
    "gpt2-large": {
        "hf_id": "gpt2-large",
        "type": "gpt2",
        "layers": 36,
        "hidden_size": 1280,
        "requires_auth": False,
        "gpu": "A10G",
    },
    "gpt2-xl": {
        "hf_id": "gpt2-xl",
        "type": "gpt2",
        "layers": 48,
        "hidden_size": 1600,
        "requires_auth": False,
        "gpu": "A10G",
    },
    # LLaMA family
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "type": "llama",
        "layers": 32,
        "hidden_size": 4096,
        "requires_auth": True,
        "gpu": "A100-80GB",
    },
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "llama",
        "layers": 22,
        "hidden_size": 2048,
        "requires_auth": False,
        "gpu": "T4",
    },
    # Mistral family
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "type": "mistral",
        "layers": 32,
        "hidden_size": 4096,
        "requires_auth": False,
        "gpu": "A100-40GB",
    },
}


DEFAULT_MODEL = "llama-3.1-8b"

# Supported datasets for evaluation
DATASETS = ["wikitext2", "c4", "ptb"]

# GPU types for hardware sweep
GPU_TYPES = ["T4", "A100-80GB", "H100"]

# ICML publication-quality configuration
# Note: Only includes modes with vLLM kv_cache_dtype mappings
ICML_CONFIG = {
    "models": ["gpt2", "gpt2-medium", "gpt2-large", "llama-3.1-8b", "mistral-7b"],
    "datasets": ["wikitext2", "c4", "ptb"],
    "cache_modes": ["fp16", "int4-hamming", "int4-hamming84",
                    "int12-golay", "int4-golay-hybrid"],
    "ber_levels": [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    "seeds": [42],
    "max_samples": 100,
    "max_length": 512,
    "stride": 256,
    "downstream_tasks": ["mmlu", "hellaswag"],
}


CACHE_MODE_LABELS = {
    "fp16": "FP16 (Oracle)",
    "int4-hamming": "Hamming(7,4)",
    "int4-hamming84": "Hamming(8,4)",
    "int12-golay": "Golay(24,12)",
    "int4-golay-hybrid": "Golay+Hamming Hybrid",
    "int4-reed-solomon": "Reed-Solomon(12,8)",
}


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


def get_ber_levels():
    """Return publication-quality BER levels for comprehensive evaluation."""
    return BER_LEVELS.copy()


def get_seeds():
    """Return seeds for multi-trial evaluation."""
    return DEFAULT_CONFIG["seeds"].copy()


def validate_evaluation_config(
    max_samples: int,
    seeds: list,
    max_length: int,
    strict: bool = True,
) -> None:
    """Validate evaluation configuration meets minimum requirements.

    Args:
        max_samples: Number of evaluation samples.
        seeds: List of random seeds for trials.
        max_length: Maximum sequence length.
        strict: If True, raise error; if False, print warning.

    Raises:
        ValueError: If configuration doesn't meet minimum requirements (strict=True).
    """
    errors = []

    if max_samples < MIN_SAMPLES:
        errors.append(
            f"max_samples={max_samples} is below minimum {MIN_SAMPLES} for statistical significance"
        )

    if len(seeds) < MIN_SEEDS:
        errors.append(
            f"len(seeds)={len(seeds)} is below minimum {MIN_SEEDS} for robust statistics"
        )

    if max_length < MIN_MAX_LENGTH:
        errors.append(
            f"max_length={max_length} is below minimum {MIN_MAX_LENGTH} for proper perplexity"
        )

    if errors:
        msg = "Evaluation configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        if strict:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")
