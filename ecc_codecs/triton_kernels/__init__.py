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
    DecodeResult,
    GolayDecodeResult,
    HAMMING74_G,
    HAMMING74_H,
    HAMMING84_G,
    HAMMING84_H,
    GOLAY_B_MATRIX,
)

from .hamming74_triton import (
    Hamming74,
    hamming74_encode,
    hamming74_decode,
)

from .hamming84_triton import (
    Hamming84,
    hamming84_encode,
    hamming84_decode,
)

from .golay_triton import (
    Golay2412,
    golay_encode,
    golay_decode,
)

from .fault_injection_triton import (
    inject_bit_errors_triton,
    inject_bit_errors_triton_batched,
    inject_bit_errors_triton_vectorized,
)

from .interpolation_triton import (
    interpolate_double_errors,
    interpolate_double_errors_1d,
    interpolate_double_errors_autotuned,
)

from .fused_kernels import (
    fused_quantize_encode_hamming84,
    fused_quantize_encode_hamming74,
    fused_decode_dequantize_hamming84,
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
    "DecodeResult",
    "GolayDecodeResult",
    "HAMMING74_G",
    "HAMMING74_H",
    "HAMMING84_G",
    "HAMMING84_H",
    "GOLAY_B_MATRIX",
    # Codec wrapper classes
    "Hamming74",
    "Hamming84",
    "Golay2412",
    # Raw functions
    "hamming74_encode",
    "hamming74_decode",
    "hamming84_encode",
    "hamming84_decode",
    "golay_encode",
    "golay_decode",
    "inject_bit_errors_triton",
    "inject_bit_errors_triton_batched",
    "inject_bit_errors_triton_vectorized",
    "interpolate_double_errors",
    "interpolate_double_errors_1d",
    "interpolate_double_errors_autotuned",
    # Fused kernels
    "fused_quantize_encode_hamming84",
    "fused_quantize_encode_hamming74",
    "fused_decode_dequantize_hamming84",
]
