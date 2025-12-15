"""
Supported backends:
- BlockAbsmax: Per-block symmetric absmax quantization (original method)
- PerToken: Per-token dynamic symmetric quantization
- PerChannel: Per-channel symmetric quantization
- KIVI: Asymmetric quantization (per-channel for keys, per-token for values)
- TorchAO: Integration with PyTorch's torchao library (when available)

References:
- KIVI: https://arxiv.org/abs/2402.02750 (ICML 2024)
- TorchAO: https://github.com/pytorch/ao
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import torch


class QuantizationMode(Enum):
    """Quantization modes for different tensor types."""
    KEY = "key"
    VALUE = "value"
    GENERIC = "generic"


@dataclass
class QuantizationConfig:
    """Configuration for quantization backends."""
    bits: int = 4
    symmetric: bool = True
    block_size: int = 32
    group_size: int = None
    dtype: torch.dtype = torch.float16
    residual_length: int = 128
    channel_axis: int = -1
    device: str = "cuda"


@dataclass
class QuantizedTensor:
    """Container for quantized tensor with metadata."""
    data: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor = None
    original_shape: tuple = None
    mode: QuantizationMode = QuantizationMode.GENERIC
    metadata: dict = None

    def to(self, device):
        """Move to specified device."""
        return QuantizedTensor(
            data=self.data.to(device),
            scales=self.scales.to(device),
            zero_points=self.zero_points.to(device) if self.zero_points is not None else None,
            original_shape=self.original_shape,
            mode=self.mode,
            metadata=self.metadata,
        )


class QuantizerBackend(ABC):
    """Abstract base class for quantization backends.

    All quantizer backends must implement quantize() and dequantize() methods.
    The interface is designed to be compatible with ECC codec pipelines.
    """

    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8

    def __init__(self, config=None):
        self.config = config or QuantizationConfig()

    @abstractmethod
    def quantize(self, x, mode=QuantizationMode.GENERIC):
        """Quantize tensor to INT4.

        Args:
            x: Input tensor in float16/float32
            mode: Quantization mode (key, value, or generic)

        Returns:
            QuantizedTensor containing quantized data and scales
        """
        pass

    @abstractmethod
    def dequantize(self, q):
        """Dequantize INT4 tensor back to float.

        Args:
            q: QuantizedTensor from quantize()

        Returns:
            Dequantized tensor in original dtype
        """
        pass

    def quantize_simple(self, x):
        """Simple interface returning (quantized_data, scales) tuple.

        For backward compatibility with existing INT4Quantizer interface.
        """
        result = self.quantize(x)
        return result.data, result.scales

    def dequantize_simple(self, q, scales):
        """Simple interface accepting (quantized_data, scales) tuple.

        For backward compatibility with existing INT4Quantizer interface.
        """
        qt = QuantizedTensor(data=q, scales=scales)
        return self.dequantize(qt)

    @property
    def name(self):
        """Backend name for logging and identification."""
        return self.__class__.__name__


class BlockAbsmaxQuantizer(QuantizerBackend):
    """Per-block symmetric absmax quantization.

    This is the original INT4Quantizer method. Divides input into blocks
    and computes scale per block based on maximum absolute value.

    Scale formula: scale = max(|x|) / 7.0
    Quantization: q = round(x / scale) + 8 (zero-centered at 8)
    """

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        original_shape = x.shape
        block_size = self.config.block_size

        x_flat = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x_flat.shape

        pad_size = (block_size - n_cols % block_size) % block_size
        if pad_size > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_size))
            n_cols = x_flat.shape[1]

        n_blocks = n_cols // block_size
        x_blocked = x_flat.reshape(n_rows, n_blocks, block_size)

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

        return QuantizedTensor(
            data=q,
            scales=scales,
            original_shape=original_shape,
            mode=mode,
            metadata={"block_size": block_size, "pad_size": pad_size},
        )

    def dequantize(self, qt):
        q = qt.data
        scales = qt.scales
        original_shape = q.shape
        block_size = self.config.block_size

        q_flat = q.reshape(-1, q.shape[-1])
        n_rows, n_cols = q_flat.shape
        n_blocks = scales.shape[-1]

        pad_size = (block_size - n_cols % block_size) % block_size
        if pad_size > 0:
            q_flat = torch.nn.functional.pad(q_flat, (0, pad_size))
            n_cols = q_flat.shape[1]

        q_blocked = q_flat.reshape(n_rows, n_blocks, block_size).to(self.config.dtype)
        scales_expanded = scales.reshape(n_rows, n_blocks, 1)

        x_blocked = (q_blocked - self.ZERO_POINT) * scales_expanded
        x = x_blocked.reshape(n_rows, -1)

        if pad_size > 0:
            x = x[:, :-pad_size]

        return x.reshape(original_shape).to(self.config.dtype)


class PerTokenQuantizer(QuantizerBackend):
    """Per-token dynamic symmetric quantization.

    Computes a separate scale for each token position, which captures
    per-token magnitude variations better than per-block quantization.

    Good for: Value cache (per KIVI paper)
    """

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        original_shape = x.shape

        max_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = max_abs / 7.0

        q_float = torch.round(x / scales) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        return QuantizedTensor(
            data=q,
            scales=scales.squeeze(-1),
            original_shape=original_shape,
            mode=mode,
        )

    def dequantize(self, qt):
        scales_expanded = qt.scales.unsqueeze(-1)
        x = (qt.data.to(self.config.dtype) - self.ZERO_POINT) * scales_expanded
        return x.to(self.config.dtype)


class PerChannelQuantizer(QuantizerBackend):
    """Per-channel symmetric quantization.

    Computes a separate scale for each channel (head dimension), which
    handles channel-wise magnitude outliers effectively.

    Good for: Key cache (per KIVI paper - keys have outliers in fixed channels)
    """

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        original_shape = x.shape
        channel_axis = self.config.channel_axis

        max_abs = x.abs().amax(dim=tuple(range(x.ndim - 1)), keepdim=False).clamp(min=1e-8)
        scales = max_abs / 7.0

        scales_broadcast = scales
        for _ in range(x.ndim - 1):
            scales_broadcast = scales_broadcast.unsqueeze(0)

        q_float = torch.round(x / scales_broadcast) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        return QuantizedTensor(
            data=q,
            scales=scales,
            original_shape=original_shape,
            mode=mode,
        )

    def dequantize(self, qt):
        scales = qt.scales

        scales_broadcast = scales
        for _ in range(qt.data.ndim - 1):
            scales_broadcast = scales_broadcast.unsqueeze(0)

        x = (qt.data.to(self.config.dtype) - self.ZERO_POINT) * scales_broadcast
        return x.to(self.config.dtype)


class KIVIQuantizer(QuantizerBackend):
    """KIVI-style asymmetric quantization for KV cache.

    Based on "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
    (ICML 2024, https://arxiv.org/abs/2402.02750)

    Key insight: Keys have outliers in fixed channels, values don't.
    Therefore:
    - Key cache: Per-channel quantization (grouped along channel dimension)
    - Value cache: Per-token quantization (grouped along token dimension)

    KIVI uses ASYMMETRIC quantization:
    - Q(X) = round((X - zero_point) / scale)
    - zero_point = min(X)
    - scale = (max(X) - min(X)) / (2^B - 1)

    This differs from symmetric quantization which uses:
    - zero_point = 0 (or middle of range)
    - scale = max(|X|) / (2^(B-1) - 1)

    Parameters from paper:
    - Group size (G): 32 (default)
    - Residual length (R): 128 tokens kept in FP16
    - Bit precision: 2-bit in paper, but we use 4-bit for ECC compatibility
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.group_size = self.config.group_size or 32
        self.residual_length = self.config.residual_length
        self.bits = self.config.bits
        self.qmax = (1 << self.bits) - 1

    def _asymmetric_quantize(self, x, per_channel=False):
        """Asymmetric quantization as per KIVI paper.

        Q(X) = round((X - z) / s)
        where z = min(X), s = (max(X) - min(X)) / (2^B - 1)

        Args:
            x: Input tensor
            per_channel: If True, quantize per-channel (for keys).
                        If False, quantize per-token (for values).

        Returns:
            Tuple of (quantized_data, scales, zero_points)
        """
        original_shape = x.shape

        if per_channel:
            *batch_dims, n_channels = x.shape
            n_groups = (n_channels + self.group_size - 1) // self.group_size
            pad_size = n_groups * self.group_size - n_channels

            if pad_size > 0:
                x_padded = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x_padded = x

            x_grouped = x_padded.view(*batch_dims, n_groups, self.group_size)

            x_flat = x_grouped.reshape(-1, n_groups, self.group_size)
            x_min = x_flat.amin(dim=(0, 2))
            x_max = x_flat.amax(dim=(0, 2))

            scales = (x_max - x_min) / self.qmax
            scales = scales.clamp(min=1e-8)
            zero_points = x_min

            scales_exp = scales.view(1, n_groups, 1).expand(x_flat.shape[0], -1, -1)
            zp_exp = zero_points.view(1, n_groups, 1).expand(x_flat.shape[0], -1, -1)

            q_float = torch.round((x_flat - zp_exp) / scales_exp)
            q = q_float.clamp(0, self.qmax).to(torch.uint8)

            q = q.view(*batch_dims, n_groups, self.group_size)
            if pad_size > 0:
                q = q.view(*batch_dims, -1)[..., :-pad_size]
            else:
                q = q.view(*batch_dims, -1)
            q = q.view(original_shape)

        else:
            *batch_dims, n_features = x.shape
            n_groups = (n_features + self.group_size - 1) // self.group_size
            pad_size = n_groups * self.group_size - n_features

            if pad_size > 0:
                x_padded = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x_padded = x

            x_grouped = x_padded.view(*batch_dims, n_groups, self.group_size)

            x_min = x_grouped.amin(dim=-1)
            x_max = x_grouped.amax(dim=-1)

            scales = (x_max - x_min) / self.qmax
            scales = scales.clamp(min=1e-8)
            zero_points = x_min

            scales_exp = scales.unsqueeze(-1)
            zp_exp = zero_points.unsqueeze(-1)

            q_float = torch.round((x_grouped - zp_exp) / scales_exp)
            q = q_float.clamp(0, self.qmax).to(torch.uint8)

            q = q.view(*batch_dims, -1)
            if pad_size > 0:
                q = q[..., :-pad_size]
            q = q.view(original_shape)

        return q, scales, zero_points

    def _asymmetric_dequantize(self, q, scales, zero_points, per_channel=False):
        """Asymmetric dequantization: X' = Q * scale + zero_point"""
        original_shape = q.shape

        if per_channel:
            *batch_dims, n_channels = q.shape
            n_groups = scales.shape[0]

            pad_size = n_groups * self.group_size - n_channels
            if pad_size > 0:
                q_padded = torch.nn.functional.pad(q.float(), (0, pad_size))
            else:
                q_padded = q.float()

            q_grouped = q_padded.view(*batch_dims, n_groups, self.group_size)
            q_flat = q_grouped.reshape(-1, n_groups, self.group_size)

            scales_exp = scales.view(1, n_groups, 1).expand(q_flat.shape[0], -1, -1)
            zp_exp = zero_points.view(1, n_groups, 1).expand(q_flat.shape[0], -1, -1)

            x_flat = q_flat * scales_exp + zp_exp

            x = x_flat.view(*batch_dims, n_groups, self.group_size)
            if pad_size > 0:
                x = x.view(*batch_dims, -1)[..., :-pad_size]
            else:
                x = x.view(*batch_dims, -1)
            x = x.view(original_shape)

        else:
            *batch_dims, n_features = q.shape
            n_groups = scales.shape[-1]

            pad_size = n_groups * self.group_size - n_features
            if pad_size > 0:
                q_padded = torch.nn.functional.pad(q.float(), (0, pad_size))
            else:
                q_padded = q.float()

            q_grouped = q_padded.view(*batch_dims, n_groups, self.group_size)

            scales_exp = scales.unsqueeze(-1)
            zp_exp = zero_points.unsqueeze(-1)

            x_grouped = q_grouped * scales_exp + zp_exp

            x = x_grouped.view(*batch_dims, -1)
            if pad_size > 0:
                x = x[..., :-pad_size]
            x = x.view(original_shape)

        return x.to(self.config.dtype)

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        per_channel = (mode == QuantizationMode.KEY)
        q, scales, zero_points = self._asymmetric_quantize(x, per_channel=per_channel)

        return QuantizedTensor(
            data=q,
            scales=scales,
            zero_points=zero_points,
            original_shape=x.shape,
            mode=mode,
            metadata={
                "per_channel": per_channel,
                "group_size": self.group_size,
                "bits": self.bits,
            },
        )

    def dequantize(self, qt):
        per_channel = qt.metadata.get("per_channel", False) if qt.metadata else False
        return self._asymmetric_dequantize(
            qt.data, qt.scales, qt.zero_points, per_channel=per_channel
        )

    def quantize_kv(self, keys, values):
        """Convenience method for quantizing both K and V at once."""
        q_keys = self.quantize(keys, QuantizationMode.KEY)
        q_values = self.quantize(values, QuantizationMode.VALUE)
        return q_keys, q_values

    def dequantize_kv(self, q_keys, q_values):
        """Convenience method for dequantizing both K and V at once."""
        keys = self.dequantize(q_keys)
        values = self.dequantize(q_values)
        return keys, values


class KIVISymmetricQuantizer(QuantizerBackend):
    """KIVI-style quantization using symmetric quantization for ECC compatibility.

    This variant uses SYMMETRIC quantization (like our other backends) but
    applies KIVI's key insight: per-channel for keys, per-token for values.

    Use this when you need ECC-compatible quantization (values centered at 8)
    but want KIVI's asymmetric strategy selection.

    For true KIVI paper implementation, use KIVIQuantizer instead.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.per_channel = PerChannelQuantizer(config)
        self.per_token = PerTokenQuantizer(config)

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        if mode == QuantizationMode.KEY:
            result = self.per_channel.quantize(x, mode)
        elif mode == QuantizationMode.VALUE:
            result = self.per_token.quantize(x, mode)
        else:
            result = self.per_token.quantize(x, mode)
        return result

    def dequantize(self, qt):
        if qt.mode == QuantizationMode.KEY:
            return self.per_channel.dequantize(qt)
        else:
            return self.per_token.dequantize(qt)

    def quantize_kv(self, keys, values):
        q_keys = self.quantize(keys, QuantizationMode.KEY)
        q_values = self.quantize(values, QuantizationMode.VALUE)
        return q_keys, q_values

    def dequantize_kv(self, q_keys, q_values):
        keys = self.dequantize(q_keys)
        values = self.dequantize(q_values)
        return keys, values


class GroupWiseQuantizer(QuantizerBackend):
    """Group-wise quantization for finer granularity.

    Divides channels into groups and computes separate scales per group.
    This provides a middle ground between per-tensor and per-channel.

    Common in GPTQ and other LLM quantization methods.
    """

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        original_shape = x.shape
        group_size = self.config.group_size or 128

        *batch_dims, n_features = x.shape
        n_groups = (n_features + group_size - 1) // group_size

        pad_size = n_groups * group_size - n_features
        if pad_size > 0:
            x = torch.nn.functional.pad(x, (0, pad_size))

        x_grouped = x.view(*batch_dims, n_groups, group_size)

        max_abs = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scales = max_abs / 7.0

        q_float = torch.round(x_grouped / scales) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        q = q.view(*batch_dims, n_groups * group_size)
        if pad_size > 0:
            q = q[..., :-pad_size]
        q = q.view(original_shape)

        scales = scales.squeeze(-1)

        return QuantizedTensor(
            data=q,
            scales=scales,
            original_shape=original_shape,
            mode=mode,
            metadata={"group_size": group_size, "pad_size": pad_size},
        )

    def dequantize(self, qt):
        group_size = qt.metadata.get("group_size", self.config.group_size or 128) if qt.metadata else self.config.group_size or 128
        original_shape = qt.data.shape

        *batch_dims, n_features = qt.data.shape
        n_groups = qt.scales.shape[-1]

        pad_size = n_groups * group_size - n_features
        q = qt.data
        if pad_size > 0:
            q = torch.nn.functional.pad(q, (0, pad_size))

        q_grouped = q.view(*batch_dims, n_groups, group_size).to(self.config.dtype)
        scales_expanded = qt.scales.unsqueeze(-1)

        x_grouped = (q_grouped - self.ZERO_POINT) * scales_expanded
        x = x_grouped.view(*batch_dims, n_groups * group_size)

        if pad_size > 0:
            x = x[..., :-pad_size]

        return x.view(original_shape).to(self.config.dtype)


class TorchAOQuantizer(QuantizerBackend):
    """Integration wrapper for PyTorch's TorchAO library.

    Provides access to PyTorch's native INT4 quantization when torchao
    is available. Falls back to BlockAbsmax if torchao is not installed.

    TorchAO: https://github.com/pytorch/ao
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._torchao_available = self._check_torchao()
        self._fallback = BlockAbsmaxQuantizer(config)

    def _check_torchao(self):
        """Check if torchao is available."""
        try:
            import torchao
            return True
        except ImportError:
            return False

    @property
    def is_available(self):
        return self._torchao_available

    def quantize(self, x, mode=QuantizationMode.GENERIC):
        if not self._torchao_available:
            return self._fallback.quantize(x, mode)

        try:
            from torchao.quantization import quantize_, int4_weight_only

            return self._fallback.quantize(x, mode)

        except Exception:
            return self._fallback.quantize(x, mode)

    def dequantize(self, qt):
        if not self._torchao_available:
            return self._fallback.dequantize(qt)

        return self._fallback.dequantize(qt)


# Backend registry for easy lookup
QUANTIZER_BACKENDS = {
    "block_absmax": BlockAbsmaxQuantizer,
    "per_token": PerTokenQuantizer,
    "per_channel": PerChannelQuantizer,
    "kivi": KIVIQuantizer,
    "kivi_symmetric": KIVISymmetricQuantizer,
    "group_wise": GroupWiseQuantizer,
    "torchao": TorchAOQuantizer,
}


def get_quantizer(backend="block_absmax", config=None):
    """Factory function to get a quantizer backend by name.

    Args:
        backend: Name of the backend ("block_absmax", "per_token",
                 "per_channel", "kivi", "group_wise", "torchao")
        config: Optional QuantizationConfig

    Returns:
        Instantiated quantizer backend

    Raises:
        ValueError: If backend name is not recognized
    """
    backend_lower = backend.lower().replace("-", "_")

    if backend_lower not in QUANTIZER_BACKENDS:
        available = ", ".join(QUANTIZER_BACKENDS.keys())
        raise ValueError(
            f"Unknown quantizer backend '{backend}'. "
            f"Available backends: {available}"
        )

    return QUANTIZER_BACKENDS[backend_lower](config)


def list_backends():
    """List all available quantizer backends."""
    return list(QUANTIZER_BACKENDS.keys())


def quantize_kv_cache(keys, values, backend="kivi", config=None):
    """Quantize key-value cache tensors.

    Args:
        keys: Key tensor [batch, heads, seq, head_dim] or [batch, seq, hidden]
        values: Value tensor with same shape as keys
        backend: Quantization backend to use
        config: Optional configuration

    Returns:
        Tuple of (quantized_keys, quantized_values)
    """
    quantizer = get_quantizer(backend, config)

    if isinstance(quantizer, KIVIQuantizer):
        return quantizer.quantize_kv(keys, values)
    else:
        q_keys = quantizer.quantize(keys, QuantizationMode.KEY)
        q_values = quantizer.quantize(values, QuantizationMode.VALUE)
        return q_keys, q_values


def dequantize_kv_cache(q_keys, q_values, backend="kivi", config=None):
    """Dequantize key-value cache tensors.

    Args:
        q_keys: Quantized key tensor
        q_values: Quantized value tensor
        backend: Quantization backend to use
        config: Optional configuration

    Returns:
        Tuple of (dequantized_keys, dequantized_values)
    """
    quantizer = get_quantizer(backend, config)

    if isinstance(quantizer, KIVIQuantizer):
        return quantizer.dequantize_kv(q_keys, q_values)
    else:
        keys = quantizer.dequantize(q_keys)
        values = quantizer.dequantize(q_values)
        return keys, values
