import torch
from typing import Tuple, Optional


class INT4Quantizer:
    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8

    def __init__(
        self,
        block_size: int = 32,
        dtype: torch.dtype = torch.float16,
    ):
        self.block_size = block_size
        self.dtype = dtype

    def quantize(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        device = x.device

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

    def dequantize(
        self,
        q: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = q.shape
        device = q.device

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


class INT4QuantizerSimple:
    QMIN = 0
    QMAX = 15
    ZERO_POINT = 8

    def __init__(self, dtype: torch.dtype = torch.float16):
        self.dtype = dtype

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_abs = x.abs().max().clamp(min=1e-8)
        scale = max_abs / 7.0

        q_float = torch.round(x / scale) + self.ZERO_POINT
        q = q_float.clamp(self.QMIN, self.QMAX).to(torch.uint8)

        return q, scale.unsqueeze(0)

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return ((q.to(self.dtype) - self.ZERO_POINT) * scale).to(self.dtype)


if __name__ == "__main__":
    print("INT4 Quantization Test")
    print("=" * 50)

    quantizer = INT4Quantizer(block_size=32)

    x = torch.randn(2, 4, 64, dtype=torch.float16)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")

    q, scales = quantizer.quantize(x)
    print(f"\nQuantized shape: {q.shape}")
    print(f"Scales shape: {scales.shape}")
    print(f"Quantized range: [{q.min()}, {q.max()}]")

    x_recon = quantizer.dequantize(q, scales)
    print(f"\nReconstructed shape: {x_recon.shape}")
    print(f"Reconstructed range: [{x_recon.min():.4f}, {x_recon.max():.4f}]")

    mse = ((x - x_recon) ** 2).mean()
    max_err = (x - x_recon).abs().max()
    print(f"\nMSE: {mse:.6f}")
    print(f"Max error: {max_err:.4f}")
