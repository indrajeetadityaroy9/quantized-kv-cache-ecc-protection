"""
Triton GPU Kernels for Error-Correcting Codes.

This package provides GPU-native implementations of:
- Hamming(7,4) SEC encode/decode
- Hamming(8,4) SECDED encode/decode
- Golay(24,12) encode/decode with shared memory LUT
- Deterministic fault injection (Philox RNG)
- Linear interpolation for double-error recovery

These kernels eliminate the CPU-GPU transfer overhead of the reference
implementations, achieving 40-50x speedup on A100 GPUs.
"""

from .config import (
    get_physical_dtype,
    get_codeword_bits,
    get_data_bits,
    HAMMING74_BLOCK_SIZE,
    HAMMING84_BLOCK_SIZE,
    GOLAY_BLOCK_SIZE,
    FAULT_INJECTION_BLOCK_SIZE,
    INTERPOLATION_BLOCK_SIZE,
    SYNDROME_LUT_HAMMING74,
    SYNDROME_LUT_HAMMING84,
    ErrorType,
)

from .hamming74_triton import (
    hamming74_encode,
    hamming74_decode,
)

from .hamming84_triton import (
    hamming84_encode,
    hamming84_decode,
)

from .golay_triton import (
    golay_encode,
    golay_decode,
)

from .fault_injection_triton import (
    inject_bit_errors_triton,
    inject_bit_errors_triton_batched,
)

from .interpolation_triton import (
    interpolate_double_errors,
    interpolate_double_errors_1d,
)

__all__ = [
    # Config
    "get_physical_dtype",
    "get_codeword_bits",
    "get_data_bits",
    "HAMMING74_BLOCK_SIZE",
    "HAMMING84_BLOCK_SIZE",
    "GOLAY_BLOCK_SIZE",
    "FAULT_INJECTION_BLOCK_SIZE",
    "INTERPOLATION_BLOCK_SIZE",
    "SYNDROME_LUT_HAMMING74",
    "SYNDROME_LUT_HAMMING84",
    "ErrorType",
    # Hamming(7,4)
    "hamming74_encode",
    "hamming74_decode",
    # Hamming(8,4)
    "hamming84_encode",
    "hamming84_decode",
    # Golay(24,12)
    "golay_encode",
    "golay_decode",
    # Fault injection
    "inject_bit_errors_triton",
    "inject_bit_errors_triton_batched",
    # Interpolation
    "interpolate_double_errors",
    "interpolate_double_errors_1d",
]
