"""
Shared Constants for Evaluation Module.

Centralizes all configuration values used across experiments and runners
to ensure consistency and eliminate configuration drift.
"""

from typing import Dict, List, Any

# =============================================================================
# Cache Modes
# =============================================================================

CACHE_MODES: Dict[str, Dict[str, Any]] = {
    "fp16": {
        "bits": 16,
        "description": "FP16 Oracle baseline (no quantization)",
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
        "bits": 8,  # 24 bits / 3 values = 8 bits per value
        "description": "INT4 triplets + Golay(24,12) - corrects up to 3 errors",
        "protected": True,
    },
    "adaptive-uep": {
        "bits": None,  # Variable: Golay for sinks (~8), Hamming for context (~8)
        "description": "Adaptive UEP: Golay(24,12) sinks + Hamming(8,4) context",
        "protected": True,
        "adaptive": True,
    },
}

# Ordered list for consistent iteration
CACHE_MODE_ORDER: List[str] = [
    "fp16",
    "int4",
    "int4-hamming",
    "int4-hamming84",
    "int4-hamming84-interp",
    "int12-golay",
    "adaptive-uep",
]

# =============================================================================
# BER Levels
# =============================================================================

# Standard BER levels for experiments
BER_LEVELS: List[float] = [0, 1e-4, 1e-3, 1e-2]

# Extended BER levels for detailed analysis
BER_LEVELS_EXTENDED: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# =============================================================================
# Default Hyperparameters
# =============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    # Perplexity computation
    "max_length": 256,
    "stride": 128,

    # Quantization
    "block_size": 32,

    # Experiment settings
    "max_samples": 50,

    # Monte Carlo seeds (5 seeds for proper statistical analysis)
    "seeds": [42, 101, 997, 1999, 4999],
}

# =============================================================================
# Model Registry
# =============================================================================

MODELS: Dict[str, Dict[str, Any]] = {
    "gpt2": {
        "hf_id": "gpt2",
        "type": "gpt2",
        "layers": 12,
        "hidden_size": 768,
        "requires_auth": False,
        "gpu": "T4",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "type": "llama",
        "layers": 32,
        "hidden_size": 4096,
        "requires_auth": True,
        "gpu": "A100-80GB",
    },
}

# Default model for experiments
DEFAULT_MODEL: str = "gpt2"

# =============================================================================
# Output Formatting
# =============================================================================

# Short labels for tables
CACHE_MODE_LABELS: Dict[str, str] = {
    "fp16": "FP16 (Oracle)",
    "int4": "INT4 (Unprotected)",
    "int4-hamming": "Hamming(7,4)",
    "int4-hamming84": "Hamming(8,4)",
    "int4-hamming84-interp": "H(8,4)+Interp",
    "int12-golay": "Golay(24,12)",
    "adaptive-uep": "Adaptive UEP",
}

# =============================================================================
# GPU Bandwidth Constants (for efficiency analysis)
# =============================================================================

# Theoretical peak memory bandwidth in GB/s for common GPU types
# Used to compute bandwidth efficiency in timing benchmarks
GPU_BANDWIDTH_GBPS: Dict[str, float] = {
    "T4": 320.0,           # NVIDIA T4 (GDDR6)
    "V100": 900.0,         # NVIDIA V100 (HBM2)
    "A100-40GB": 1555.0,   # NVIDIA A100 40GB (HBM2e)
    "A100-80GB": 2039.0,   # NVIDIA A100 80GB (HBM2e)
    "A10G": 600.0,         # NVIDIA A10G (GDDR6X)
    "L4": 300.0,           # NVIDIA L4 (GDDR6)
    "H100": 3352.0,        # NVIDIA H100 (HBM3)
    "RTX3090": 936.2,      # NVIDIA RTX 3090 (GDDR6X)
    "RTX4090": 1008.0,     # NVIDIA RTX 4090 (GDDR6X)
}

# Default GPU for bandwidth calculations
DEFAULT_GPU: str = "A100-80GB"


def get_gpu_bandwidth(gpu_type: str = None) -> float:
    """Get theoretical peak bandwidth for GPU type.

    Args:
        gpu_type: GPU type string (e.g., "A100-80GB"). Uses DEFAULT_GPU if None.

    Returns:
        Theoretical peak memory bandwidth in GB/s
    """
    if gpu_type is None:
        gpu_type = DEFAULT_GPU
    return GPU_BANDWIDTH_GBPS.get(gpu_type, GPU_BANDWIDTH_GBPS[DEFAULT_GPU])


def compute_bandwidth_efficiency(
    throughput_mvalues_sec: float,
    bytes_per_value: int = 1,
    gpu_type: str = None,
) -> float:
    """Compute bandwidth efficiency as percentage of theoretical peak.

    Args:
        throughput_mvalues_sec: Measured throughput in millions of values/sec
        bytes_per_value: Bytes per value (1 for uint8, 2 for fp16, etc.)
        gpu_type: GPU type for theoretical peak lookup

    Returns:
        Bandwidth efficiency as percentage (0-100)
    """
    peak_bandwidth = get_gpu_bandwidth(gpu_type)
    # Convert throughput to GB/s: M values/sec * bytes/value / 1000 = GB/s
    achieved_bandwidth = throughput_mvalues_sec * bytes_per_value / 1000.0
    return 100.0 * achieved_bandwidth / peak_bandwidth


# =============================================================================
# Helper Functions
# =============================================================================

def get_cache_modes(protected_only: bool = False) -> List[str]:
    """Get list of cache modes, optionally filtering to protected only."""
    if protected_only:
        return [m for m in CACHE_MODE_ORDER if CACHE_MODES[m]["protected"]]
    return CACHE_MODE_ORDER.copy()


def get_ber_levels(extended: bool = False) -> List[float]:
    """Get BER levels for experiments."""
    if extended:
        return BER_LEVELS_EXTENDED.copy()
    return BER_LEVELS.copy()


def get_seeds() -> List[int]:
    """Get random seeds for Monte Carlo experiments."""
    return DEFAULT_CONFIG["seeds"].copy()
