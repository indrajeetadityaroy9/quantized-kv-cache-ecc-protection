_lazy_imports_done = False
_quantization_imports_done = False
_module_cache = {}


def _do_lazy_imports():
    global _lazy_imports_done, _module_cache
    if _lazy_imports_done:
        return

    # Import directly from config (doesn't require triton)
    from .triton_kernels.config import (
        ErrorType,
        DecodeResult,
        GolayDecodeResult,
        HAMMING74_G,
        HAMMING74_H,
        HAMMING84_G,
        HAMMING84_H,
        GOLAY_B_MATRIX,
    )

    # These require triton - only import when actually needed
    from .triton_kernels.hamming74_triton import Hamming74
    from .triton_kernels.hamming84_triton import Hamming84
    from .triton_kernels.golay_triton import Golay2412
    from .quantization import INT4Quantizer

    _module_cache.update({
        "Hamming74": Hamming74,
        "Hamming84": Hamming84,
        "ErrorType": ErrorType,
        "DecodeResult": DecodeResult,
        "Golay2412": Golay2412,
        "GolayDecodeResult": GolayDecodeResult,
        "INT4Quantizer": INT4Quantizer,
        "HAMMING74_G": HAMMING74_G,
        "HAMMING74_H": HAMMING74_H,
        "HAMMING84_G": HAMMING84_G,
        "HAMMING84_H": HAMMING84_H,
        "GOLAY_B_MATRIX": GOLAY_B_MATRIX,
    })
    _lazy_imports_done = True


def _do_quantization_imports():
    """Import quantization backends (doesn't require triton)."""
    global _quantization_imports_done, _module_cache
    if _quantization_imports_done:
        return

    from .quantization_backends import (
        QuantizerBackend,
        QuantizationConfig,
        QuantizationMode,
        QuantizedTensor,
        BlockAbsmaxQuantizer,
        PerTokenQuantizer,
        PerChannelQuantizer,
        KIVIQuantizer,
        KIVISymmetricQuantizer,
        GroupWiseQuantizer,
        TorchAOQuantizer,
        get_quantizer,
        list_backends,
        quantize_kv_cache,
        dequantize_kv_cache,
        QUANTIZER_BACKENDS,
    )

    _module_cache.update({
        "QuantizerBackend": QuantizerBackend,
        "QuantizationConfig": QuantizationConfig,
        "QuantizationMode": QuantizationMode,
        "QuantizedTensor": QuantizedTensor,
        "BlockAbsmaxQuantizer": BlockAbsmaxQuantizer,
        "PerTokenQuantizer": PerTokenQuantizer,
        "PerChannelQuantizer": PerChannelQuantizer,
        "KIVIQuantizer": KIVIQuantizer,
        "KIVISymmetricQuantizer": KIVISymmetricQuantizer,
        "GroupWiseQuantizer": GroupWiseQuantizer,
        "TorchAOQuantizer": TorchAOQuantizer,
        "get_quantizer": get_quantizer,
        "list_backends": list_backends,
        "quantize_kv_cache": quantize_kv_cache,
        "dequantize_kv_cache": dequantize_kv_cache,
        "QUANTIZER_BACKENDS": QUANTIZER_BACKENDS,
    })
    _quantization_imports_done = True


_TRITON_EXPORTS = (
    "Hamming74",
    "Hamming84",
    "ErrorType",
    "DecodeResult",
    "Golay2412",
    "GolayDecodeResult",
    "INT4Quantizer",
    "HAMMING74_G",
    "HAMMING74_H",
    "HAMMING84_G",
    "HAMMING84_H",
    "GOLAY_B_MATRIX",
)

_QUANTIZATION_EXPORTS = (
    "QuantizerBackend",
    "QuantizationConfig",
    "QuantizationMode",
    "QuantizedTensor",
    "BlockAbsmaxQuantizer",
    "PerTokenQuantizer",
    "PerChannelQuantizer",
    "KIVIQuantizer",
    "KIVISymmetricQuantizer",
    "GroupWiseQuantizer",
    "TorchAOQuantizer",
    "get_quantizer",
    "list_backends",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "QUANTIZER_BACKENDS",
)


def __getattr__(name):
    if name in _TRITON_EXPORTS:
        _do_lazy_imports()
        return _module_cache[name]
    if name in _QUANTIZATION_EXPORTS:
        _do_quantization_imports()
        return _module_cache[name]
    raise AttributeError(f"module 'hamming74' has no attribute '{name}'")


__all__ = [
    # ECC Codecs (require triton)
    "Hamming74",
    "Hamming84",
    "ErrorType",
    "DecodeResult",
    "Golay2412",
    "GolayDecodeResult",
    "HAMMING74_G",
    "HAMMING74_H",
    "HAMMING84_G",
    "HAMMING84_H",
    "GOLAY_B_MATRIX",
    # Quantization (no triton required)
    "INT4Quantizer",
    "QuantizerBackend",
    "QuantizationConfig",
    "QuantizationMode",
    "QuantizedTensor",
    "BlockAbsmaxQuantizer",
    "PerTokenQuantizer",
    "PerChannelQuantizer",
    "KIVIQuantizer",
    "KIVISymmetricQuantizer",
    "GroupWiseQuantizer",
    "TorchAOQuantizer",
    "get_quantizer",
    "list_backends",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "QUANTIZER_BACKENDS",
]
