from typing import Dict, List, Any


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
        "bits": 8,
        "description": "INT4 triplets + Golay(24,12) - corrects up to 3 errors",
        "protected": True,
    },
    "adaptive-uep": {
        "bits": None,
        "description": "Adaptive UEP: Golay(24,12) sinks + Hamming(8,4) context",
        "protected": True,
        "adaptive": True,
    },
}


CACHE_MODE_ORDER: List[str] = [
    "fp16",
    "int4",
    "int4-hamming",
    "int4-hamming84",
    "int4-hamming84-interp",
    "int12-golay",
    "adaptive-uep",
]


BER_LEVELS: List[float] = [0, 1e-4, 1e-3, 1e-2]


BER_LEVELS_EXTENDED: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


DEFAULT_CONFIG: Dict[str, Any] = {
    "max_length": 256,
    "stride": 128,
    "block_size": 32,
    "max_samples": 50,
    "seeds": [42, 101, 997, 1999, 4999],
}


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


DEFAULT_MODEL: str = "gpt2"


CACHE_MODE_LABELS: Dict[str, str] = {
    "fp16": "FP16 (Oracle)",
    "int4": "INT4 (Unprotected)",
    "int4-hamming": "Hamming(7,4)",
    "int4-hamming84": "Hamming(8,4)",
    "int4-hamming84-interp": "H(8,4)+Interp",
    "int12-golay": "Golay(24,12)",
    "adaptive-uep": "Adaptive UEP",
}


GPU_BANDWIDTH_GBPS: Dict[str, float] = {
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


DEFAULT_GPU: str = "A100-80GB"


def get_gpu_bandwidth(gpu_type: str = None) -> float:
    if gpu_type is None:
        gpu_type = DEFAULT_GPU
    return GPU_BANDWIDTH_GBPS.get(gpu_type, GPU_BANDWIDTH_GBPS[DEFAULT_GPU])


def compute_bandwidth_efficiency(
    throughput_mvalues_sec: float,
    bytes_per_value: int = 1,
    gpu_type: str = None,
) -> float:
    peak_bandwidth = get_gpu_bandwidth(gpu_type)

    achieved_bandwidth = throughput_mvalues_sec * bytes_per_value / 1000.0
    return 100.0 * achieved_bandwidth / peak_bandwidth


def get_cache_modes(protected_only: bool = False) -> List[str]:
    if protected_only:
        return [m for m in CACHE_MODE_ORDER if CACHE_MODES[m]["protected"]]
    return CACHE_MODE_ORDER.copy()


def get_ber_levels(extended: bool = False) -> List[float]:
    if extended:
        return BER_LEVELS_EXTENDED.copy()
    return BER_LEVELS.copy()


def get_seeds() -> List[int]:
    return DEFAULT_CONFIG["seeds"].copy()
