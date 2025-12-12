"""
Hamming Error Correction for INT4 Quantized LLM KV Caches.

This package provides:
- Hamming(7,4): Single-error correction (SEC), 7 bits per value
- Hamming(8,4): SECDED (Single Error Correction, Double Error Detection), 8 bits per value
- Hamming(8,4) + Interpolation: SECDED with soft erasure decoding for double errors
- Golay(24,12): Corrects up to 3 errors, bundles 3 INT4 values
- GPU model integration via the Triton/vLLM attention shim

GPU Acceleration:
- Triton kernels in hamming74.triton_kernels provide 40-50x speedup
- vllm_kernels.shim provides GPU-native model integration
"""

# Core codec implementations (still needed by Triton for verification/LUT building)
from .hamming74_sec import Hamming74
from .hamming84_secded import Hamming84, ErrorType, DecodeResult
from .golay import Golay2412, GolayDecodeResult
from .quantization import INT4Quantizer

# Legacy CPU path removed; use Triton/vLLM shim for integration.

__all__ = [
    # Hamming codecs
    "Hamming74",
    "Hamming84",
    "ErrorType",
    "DecodeResult",
    # Golay codec
    "Golay2412",
    "GolayDecodeResult",
    # Quantization
    "INT4Quantizer",
]
