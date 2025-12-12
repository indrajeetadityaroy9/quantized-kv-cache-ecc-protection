"""
INT4 Symmetric Quantization for KV Cache Values.

Quantization scheme (from proposal):
    q = clip(round(x/s + z), 0, 15)

where:
    s = scale factor (computed per block)
    z = zero point (typically 8 for symmetric)

For symmetric quantization around zero:
    - Range maps to [-8, 7] internally
    - Stored as unsigned 4-bit (0-15)
    - Zero point = 8
"""

import torch
from typing import Tuple, Optional


class INT4Quantizer:
    """
    Block-wise symmetric INT4 quantizer for KV cache values.

    Features:
    - Per-block scaling for better accuracy
    - Symmetric quantization (zero maps to 8)
    - Supports FP16/FP32 input and output
    """

    # INT4 range
    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8  # Symmetric: -8 to 7 maps to 0 to 15

    def __init__(
        self,
        block_size: int = 32,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize INT4 quantizer.

        Args:
            block_size: Number of values per quantization block
            dtype: Data type for dequantized output
        """
        self.block_size = block_size
        self.dtype = dtype

    def quantize(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize FP tensor to INT4 with per-block scaling.

        Args:
            x: Input tensor of any shape (last dim will be blocked)

        Returns:
            q: Quantized INT4 values as uint8 (0-15)
            scales: Per-block scale factors
        """
        original_shape = x.shape
        device = x.device

        # Flatten to 2D for block processing
        x_flat = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x_flat.shape

        # Pad last dimension to multiple of block_size
        pad_size = (self.block_size - n_cols % self.block_size) % self.block_size
        if pad_size > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_size))
            n_cols = x_flat.shape[1]

        # Reshape into blocks: (n_rows, n_blocks, block_size)
        n_blocks = n_cols // self.block_size
        x_blocked = x_flat.reshape(n_rows, n_blocks, self.block_size)

        # Compute per-block scale: max absolute value
        max_abs = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)

        # Scale to [-8, 7] range, then shift to [0, 15]
        # scale = max_abs / 7 (so max maps to 7, min to -7 at most)
        scales = max_abs / 7.0

        # Quantize: q = round(x/scale) + zero_point
        q_float = torch.round(x_blocked / scales) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        # Reshape back
        q = q.reshape(n_rows, n_cols)
        scales = scales.squeeze(-1)  # (n_rows, n_blocks)

        # Remove padding from quantized values
        if pad_size > 0:
            q = q[:, :-pad_size]

        # Restore original shape
        q = q.reshape(original_shape)
        scales = scales.reshape(*original_shape[:-1], n_blocks)

        return q, scales

    def dequantize(
        self,
        q: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize INT4 values back to floating point.

        Args:
            q: Quantized values (uint8, 0-15)
            scales: Per-block scale factors

        Returns:
            x: Dequantized tensor in self.dtype
        """
        original_shape = q.shape
        device = q.device

        # Flatten for block processing
        q_flat = q.reshape(-1, q.shape[-1])
        n_rows, n_cols = q_flat.shape
        n_blocks = scales.shape[-1]
        block_size = self.block_size

        # Pad to match original blocked shape
        pad_size = (block_size - n_cols % block_size) % block_size
        if pad_size > 0:
            q_flat = torch.nn.functional.pad(q_flat, (0, pad_size))
            n_cols = q_flat.shape[1]

        # Reshape into blocks
        q_blocked = q_flat.reshape(n_rows, n_blocks, block_size).to(self.dtype)

        # Expand scales for broadcasting
        scales_expanded = scales.reshape(n_rows, n_blocks, 1)

        # Dequantize: x = (q - zero_point) * scale
        x_blocked = (q_blocked - self.ZERO_POINT) * scales_expanded

        # Reshape and remove padding
        x = x_blocked.reshape(n_rows, -1)
        if pad_size > 0:
            x = x[:, :-pad_size]

        return x.reshape(original_shape).to(self.dtype)


class INT4QuantizerSimple:
    """
    Simple per-tensor INT4 quantizer (no blocking).
    Useful for testing and as a baseline.
    """

    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8

    def __init__(self, dtype: torch.dtype = torch.float16):
        self.dtype = dtype

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to INT4 with single scale factor."""
        max_abs = x.abs().max().clamp(min=1e-8)
        scale = max_abs / 7.0

        q_float = torch.round(x / scale) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        return q, scale.unsqueeze(0)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT4 values."""
        return ((q.to(self.dtype) - self.ZERO_POINT) * scale).to(self.dtype)


if __name__ == "__main__":
    # Test quantization round-trip
    print("INT4 Quantization Test")
    print("=" * 50)

    quantizer = INT4Quantizer(block_size=32)

    # Create test tensor
    x = torch.randn(2, 4, 64, dtype=torch.float16)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    # Quantize
    q, scales = quantizer.quantize(x)
    print(f"\nQuantized shape: {q.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Quantized range: [{q.min()}, {q.max()}]")

    # Dequantize
    x_recon = quantizer.dequantize(q, scales)
    print(f"\nReconstructed shape: {x_recon.shape}")
    print(f"Reconstructed range: [{x_recon.min():.4f}, {x_recon.max():.4f}]")

    # Compute error
    mse = ((x - x_recon) ** 2).mean()
    max_err = (x - x_recon).abs().max()
    print(f"\nMSE: {mse:.6f}")
    print(f"Max error: {max_err:.4f}")
