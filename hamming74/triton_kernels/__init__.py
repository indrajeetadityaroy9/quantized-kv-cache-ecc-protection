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
    "hamming74_encode",
    "hamming74_decode",
    "hamming84_encode",
    "hamming84_decode",
    "golay_encode",
    "golay_decode",
    "inject_bit_errors_triton",
    "inject_bit_errors_triton_batched",
    "interpolate_double_errors",
    "interpolate_double_errors_1d",
]
