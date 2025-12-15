"""INT4 Quantization for KV Cache with Pluggable Backends.

This module provides INT4 quantization with support for multiple quantization
strategies through a pluggable backend system.

Available backends:
- block_absmax: Per-block symmetric absmax (default, original method)
- per_token: Per-token dynamic symmetric
- per_channel: Per-channel symmetric
- kivi: KIVI-style asymmetric (per-channel keys, per-token values)
- group_wise: Group-wise symmetric
- torchao: PyTorch TorchAO integration (when available)

Example:
    # Original interface (backward compatible)
    quantizer = INT4Quantizer(block_size=32)
    q, scales = quantizer.quantize(x)
    x_restored = quantizer.dequantize(q, scales)

    # With explicit backend
    quantizer = INT4Quantizer(backend="kivi")
    q, scales = quantizer.quantize(x, mode="key")  # Per-channel for keys

    # Or use backends directly
    from hamming74.quantization_backends import get_quantizer, QuantizationMode
    kivi = get_quantizer("kivi")
    q_keys = kivi.quantize(keys, QuantizationMode.KEY)
"""

import torch


class INT4Quantizer:
    """INT4 Quantizer with pluggable backend support.

    Provides symmetric INT4 quantization [0, 15] with zero-point at 8.
    Supports multiple quantization strategies through backend parameter.

    Args:
        block_size: Block size for block-based quantization (default: 32)
        dtype: Output dtype for dequantized values (default: float16)
        backend: Quantization backend name (default: "block_absmax")
                Options: "block_absmax", "per_token", "per_channel",
                         "kivi", "group_wise", "torchao"
        group_size: Group size for group-wise quantization (default: 128)

    Attributes:
        QMIN: Minimum quantized value (0)
        QMAX: Maximum quantized value (15)
        ZERO_POINT: Zero point for symmetric quantization (8)
    """

    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8

    def __init__(self, block_size=32, dtype=torch.float16, backend="block_absmax", group_size=128):
        self.block_size = block_size
        self.dtype = dtype
        self._backend_name = backend
        self._group_size = group_size
        self._backend = None  # Lazy initialization

    @property
    def backend(self):
        """Lazy-load the backend to avoid circular imports."""
        if self._backend is None:
            from .quantization_backends import get_quantizer, QuantizationConfig

            config = QuantizationConfig(
                block_size=self.block_size,
                group_size=self._group_size,
                dtype=self.dtype,
            )
            self._backend = get_quantizer(self._backend_name, config)
        return self._backend

    @property
    def backend_name(self):
        """Get the current backend name."""
        return self._backend_name

    def quantize_full(self, x, mode=None):
        """Quantize tensor and return full QuantizedTensor result.

        Use this method for asymmetric backends (KIVI) to get zero_points.

        Args:
            x: Input tensor in float16/float32
            mode: Optional quantization mode ("key", "value", or None)

        Returns:
            QuantizedTensor with data, scales, zero_points, and metadata
        """
        from .quantization_backends import QuantizationMode, QuantizedTensor

        qmode = QuantizationMode.GENERIC
        if mode == "key":
            qmode = QuantizationMode.KEY
        elif mode == "value":
            qmode = QuantizationMode.VALUE

        result = self.backend.quantize(x, qmode)
        self._last_quantized = result
        return result

    def dequantize_full(self, qt):
        """Dequantize from full QuantizedTensor.

        Args:
            qt: QuantizedTensor from quantize_full()

        Returns:
            Dequantized tensor
        """
        return self.backend.dequantize(qt)

    def quantize(self, x, mode=None):
        """Quantize tensor to INT4.

        Args:
            x: Input tensor in float16/float32
            mode: Optional quantization mode ("key", "value", or None for generic)
                  Only used with KIVI backend for asymmetric quantization

        Returns:
            Tuple of (quantized_data, scales)
            Note: For asymmetric backends (kivi), use quantize_full() to get zero_points
        """
        # For non-block_absmax backends or when mode is specified, use backend
        if self._backend_name != "block_absmax" or mode is not None:
            from .quantization_backends import QuantizationMode

            qmode = QuantizationMode.GENERIC
            if mode == "key":
                qmode = QuantizationMode.KEY
            elif mode == "value":
                qmode = QuantizationMode.VALUE

            result = self.backend.quantize(x, qmode)
            # Cache for asymmetric dequantization
            self._last_quantized = result
            return result.data, result.scales

        # Original block_absmax implementation (fast path for backward compat)
        original_shape = x.shape

        x_flat = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x_flat.shape

        pad_size = (self.block_size - n_cols % self.block_size) % self.block_size
        if pad_size > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_size))
            n_cols = x_flat.shape[1]

        n_blocks = n_cols // self.block_size
        x_blocked = x_flat.reshape(n_rows, n_blocks, self.block_size)

        max_abs = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

        scales = max_abs / 7.0

        q_float = torch.round(x_blocked / scales) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        q = q.reshape(n_rows, n_cols)
        scales = scales.squeeze(-1)

        if pad_size > 0:
            q = q[:, :-pad_size]

        q = q.reshape(original_shape)
        scales = scales.reshape(*original_shape[:-1], n_blocks)

        return q, scales

    def dequantize(self, q, scales, mode=None, zero_points=None):
        """Dequantize INT4 tensor back to float.

        Args:
            q: Quantized tensor from quantize()
            scales: Scale tensor from quantize()
            mode: Optional quantization mode (for KIVI backend)
            zero_points: Optional zero points for asymmetric quantization (KIVI).
                        If None and using KIVI, will use cached value from last quantize().

        Returns:
            Dequantized tensor in original dtype
        """
        # For non-block_absmax backends or when mode is specified, use backend
        if self._backend_name != "block_absmax" or mode is not None:
            from .quantization_backends import QuantizedTensor, QuantizationMode

            qmode = QuantizationMode.GENERIC
            if mode == "key":
                qmode = QuantizationMode.KEY
            elif mode == "value":
                qmode = QuantizationMode.VALUE

            # For asymmetric backends, try to get zero_points
            zp = zero_points
            metadata = None
            if zp is None and hasattr(self, '_last_quantized') and self._last_quantized is not None:
                # Use cached zero_points from last quantize() call
                zp = self._last_quantized.zero_points
                metadata = self._last_quantized.metadata

            qt = QuantizedTensor(data=q, scales=scales, zero_points=zp, mode=qmode, metadata=metadata)
            return self.backend.dequantize(qt)

        # Original block_absmax implementation (fast path for backward compat)
        original_shape = q.shape

        q_flat = q.reshape(-1, q.shape[-1])
        n_rows, n_cols = q_flat.shape
        n_blocks = scales.shape[-1]
        block_size = self.block_size

        pad_size = (block_size - n_cols % block_size) % block_size
        if pad_size > 0:
            q_flat = torch.nn.functional.pad(q_flat, (0, pad_size))
            n_cols = q_flat.shape[1]

        q_blocked = q_flat.reshape(n_rows, n_blocks, block_size).to(self.dtype)

        scales_expanded = scales.reshape(n_rows, n_blocks, 1)

        x_blocked = (q_blocked - self.ZERO_POINT) * scales_expanded

        x = x_blocked.reshape(n_rows, -1)
        if pad_size > 0:
            x = x[:, :-pad_size]

        return x.reshape(original_shape).to(self.dtype)

    def quantize_kv(self, keys, values):
        """Quantize key-value pair with appropriate strategies.

        For KIVI backend: keys use per-channel, values use per-token.
        For other backends: both use the same strategy.

        Args:
            keys: Key tensor
            values: Value tensor

        Returns:
            Tuple of ((q_keys, k_scales), (q_values, v_scales))
        """
        q_keys, k_scales = self.quantize(keys, mode="key")
        q_values, v_scales = self.quantize(values, mode="value")
        return (q_keys, k_scales), (q_values, v_scales)

    def dequantize_kv(self, q_keys, k_scales, q_values, v_scales):
        """Dequantize key-value pair.

        Args:
            q_keys: Quantized keys
            k_scales: Key scales
            q_values: Quantized values
            v_scales: Value scales

        Returns:
            Tuple of (keys, values)
        """
        keys = self.dequantize(q_keys, k_scales, mode="key")
        values = self.dequantize(q_values, v_scales, mode="value")
        return keys, values


class INT4QuantizerSimple:
    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8

    def __init__(self, dtype=torch.float16):
        self.dtype = dtype

    def quantize(self, x):
        max_abs = x.abs().max().clamp(min=1e-8)
        scale = max_abs / 7.0

        q_float = torch.round(x / scale) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        return q, scale.unsqueeze(0)

    def dequantize(self, q, scale):
        return ((q.to(self.dtype) - self.ZERO_POINT) * scale).to(self.dtype)


def _test_backend(name, x, verbose=True):
    """Test a specific quantization backend."""
    quantizer = INT4Quantizer(block_size=32, backend=name)

    q, scales = quantizer.quantize(x)
    x_recon = quantizer.dequantize(q, scales)

    mse = ((x - x_recon) ** 2).mean().item()

    if verbose:
        print(f"\n{name.upper()} Backend:")
        print(f"  Quantized shape: {q.shape}, Scales shape: {scales.shape}")
        print(f"  MSE: {mse:.6f}")

    return mse


if __name__ == "__main__":
    print("INT4 Quantization Test with Pluggable Backends")
    print("=" * 60)

    # Test tensor
    x = torch.randn(2, 4, 64, dtype=torch.float16)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    # Test all backends
    from .quantization_backends import list_backends

    print("\n" + "-" * 60)
    print("Testing all backends:")
    print("-" * 60)

    results = {}
    for backend in list_backends():
        try:
            mse = _test_backend(backend, x)
            results[backend] = mse
        except Exception as e:
            print(f"\n{backend.upper()} Backend: FAILED ({e})")
            results[backend] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for backend, mse in results.items():
        status = f"MSE={mse:.6f}" if mse is not None else "FAILED"
        print(f"  {backend:<15}: {status}")

    # Test KIVI-style KV quantization
    print("\n" + "-" * 60)
    print("Testing KIVI KV Cache Quantization:")
    print("-" * 60)

    kivi_quantizer = INT4Quantizer(backend="kivi")

    # Simulate KV cache tensors
    keys = torch.randn(2, 8, 64, 32, dtype=torch.float16)  # [batch, heads, seq, head_dim]
    values = torch.randn(2, 8, 64, 32, dtype=torch.float16)

    (q_k, k_scales), (q_v, v_scales) = kivi_quantizer.quantize_kv(keys, values)
    keys_recon, values_recon = kivi_quantizer.dequantize_kv(q_k, k_scales, q_v, v_scales)

    k_mse = ((keys - keys_recon) ** 2).mean().item()
    v_mse = ((values - values_recon) ** 2).mean().item()

    print(f"  Keys:   shape={keys.shape} -> q_shape={q_k.shape}, MSE={k_mse:.6f}")
    print(f"  Values: shape={values.shape} -> q_shape={q_v.shape}, MSE={v_mse:.6f}")

    print("\nAll tests completed!")
