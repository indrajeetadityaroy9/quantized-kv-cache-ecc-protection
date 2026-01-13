"""
Fused Triton kernels for quantization and ECC encoding.

These kernels eliminate memory round-trips by combining:
- Scale computation (reduction)
- INT4 quantization (element-wise)
- ECC encoding (element-wise)

Into single GPU kernel launches.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _fused_quantize_encode_hamming84_kernel(
    input_ptr,       # [num_rows, row_size] float16/float32 input
    codeword_ptr,    # [num_rows, row_size] uint8 output
    scale_ptr,       # [num_rows] float32 scales
    num_rows,
    row_size,
    ROW_SIZE_PADDED: tl.constexpr,  # Padded to power of 2 for reduction
):
    """
    Fused kernel: scale computation + INT4 quantization + Hamming84 encode.

    Each program handles one row (e.g., one token's head_dim values).
    """
    pid = tl.program_id(0)

    if pid >= num_rows:
        return

    # Compute base pointers for this row
    row_input_ptr = input_ptr + pid * row_size
    row_output_ptr = codeword_ptr + pid * row_size

    # Load entire row (padded to power of 2 for reduction)
    offsets = tl.arange(0, ROW_SIZE_PADDED)
    mask = offsets < row_size

    # Load values as float32 for computation
    x = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Step 1: Compute scale via parallel reduction (absmax)
    abs_x = tl.abs(x)
    abs_max = tl.max(abs_x)

    # Scale = absmax / 7.0 (symmetric quantization to [-8, 7])
    scale = abs_max / 7.0
    # Avoid division by zero
    scale = tl.where(scale == 0.0, 1.0, scale)

    # Store scale
    tl.store(scale_ptr + pid, scale)

    # Step 2: Quantize to INT4 (symmetric, offset by 8 for unsigned storage)
    q = libdevice.rint(x / scale)
    q = tl.maximum(tl.minimum(q, 7.0), -8.0)
    int4_val = (q + 8.0).to(tl.uint8)

    # Step 3: Hamming84 encode inline
    d0 = (int4_val >> 0) & 1
    d1 = (int4_val >> 1) & 1
    d2 = (int4_val >> 2) & 1
    d3 = (int4_val >> 3) & 1

    p0 = d0 ^ d1 ^ d3
    p1 = d0 ^ d2 ^ d3
    p2 = d1 ^ d2 ^ d3

    hamming7 = (
        (d0 << 0)
        | (d1 << 1)
        | (d2 << 2)
        | (d3 << 3)
        | (p0 << 4)
        | (p1 << 5)
        | (p2 << 6)
    )

    # Overall parity bit (SECDED)
    parity = hamming7 ^ (hamming7 >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    overall_parity = parity & 1

    codeword = (hamming7 | (overall_parity << 7)).to(tl.uint8)

    # Store codewords
    tl.store(row_output_ptr + offsets, codeword, mask=mask)


def fused_quantize_encode_hamming84(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused quantization and Hamming84 encoding in a single kernel launch.

    Args:
        input_tensor: Input tensor of shape [..., row_size] (float16 or float32)
                     Quantization scale is computed over the last dimension.

    Returns:
        tuple: (encoded_codewords, scales)
            - encoded_codewords: uint8 tensor same shape as input
            - scales: float32 tensor of shape [...] (last dim removed)

    Example:
        >>> x = torch.randn(1000, 128, device='cuda', dtype=torch.float16)
        >>> codewords, scales = fused_quantize_encode_hamming84(x)
        >>> # codewords.shape = (1000, 128), scales.shape = (1000,)
    """
    assert input_tensor.is_cuda, "Input must be on CUDA device"

    # Handle arbitrary batch dimensions by flattening to 2D
    original_shape = input_tensor.shape
    row_size = original_shape[-1]

    if input_tensor.dim() == 1:
        # Single row
        flat_input = input_tensor.unsqueeze(0)
        num_rows = 1
    else:
        # Flatten all leading dims
        flat_input = input_tensor.reshape(-1, row_size)
        num_rows = flat_input.shape[0]

    # Ensure contiguous
    flat_input = flat_input.contiguous()

    # Allocate outputs
    codewords = torch.empty(num_rows, row_size, dtype=torch.uint8, device=input_tensor.device)
    scales = torch.empty(num_rows, dtype=torch.float32, device=input_tensor.device)

    # Compute padded row size (power of 2 for efficient reduction)
    row_size_padded = triton.next_power_of_2(row_size)

    # Launch kernel: one program per row
    grid = (num_rows,)
    _fused_quantize_encode_hamming84_kernel[grid](
        flat_input,
        codewords,
        scales,
        num_rows,
        row_size,
        ROW_SIZE_PADDED=row_size_padded,
    )

    # Reshape outputs to match input batch dims
    if input_tensor.dim() == 1:
        codewords = codewords.squeeze(0)
    else:
        codewords = codewords.view(original_shape)
        scales = scales.view(original_shape[:-1])

    return codewords, scales


@triton.jit
def _fused_quantize_encode_hamming74_kernel(
    input_ptr,       # [num_rows, row_size] float16/float32 input
    codeword_ptr,    # [num_rows, row_size] uint8 output
    scale_ptr,       # [num_rows] float32 scales
    num_rows,
    row_size,
    ROW_SIZE_PADDED: tl.constexpr,
):
    """
    Fused kernel: scale computation + INT4 quantization + Hamming74 encode.
    """
    pid = tl.program_id(0)

    if pid >= num_rows:
        return

    row_input_ptr = input_ptr + pid * row_size
    row_output_ptr = codeword_ptr + pid * row_size

    offsets = tl.arange(0, ROW_SIZE_PADDED)
    mask = offsets < row_size

    x = tl.load(row_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Scale computation
    abs_max = tl.max(tl.abs(x))
    scale = abs_max / 7.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    tl.store(scale_ptr + pid, scale)

    # Quantize
    q = libdevice.rint(x / scale)
    q = tl.maximum(tl.minimum(q, 7.0), -8.0)
    int4_val = (q + 8.0).to(tl.uint8)

    # Hamming74 encode (7 bits, no overall parity)
    d0 = (int4_val >> 0) & 1
    d1 = (int4_val >> 1) & 1
    d2 = (int4_val >> 2) & 1
    d3 = (int4_val >> 3) & 1

    p0 = d0 ^ d1 ^ d3
    p1 = d0 ^ d2 ^ d3
    p2 = d1 ^ d2 ^ d3

    codeword = (
        (d0 << 0)
        | (d1 << 1)
        | (d2 << 2)
        | (d3 << 3)
        | (p0 << 4)
        | (p1 << 5)
        | (p2 << 6)
    ).to(tl.uint8)

    tl.store(row_output_ptr + offsets, codeword, mask=mask)


def fused_quantize_encode_hamming74(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused quantization and Hamming74 encoding in a single kernel launch.

    Args:
        input_tensor: Input tensor of shape [..., row_size]

    Returns:
        tuple: (encoded_codewords, scales)
    """
    assert input_tensor.is_cuda, "Input must be on CUDA device"

    original_shape = input_tensor.shape
    row_size = original_shape[-1]

    if input_tensor.dim() == 1:
        flat_input = input_tensor.unsqueeze(0)
        num_rows = 1
    else:
        flat_input = input_tensor.reshape(-1, row_size)
        num_rows = flat_input.shape[0]

    flat_input = flat_input.contiguous()

    codewords = torch.empty(num_rows, row_size, dtype=torch.uint8, device=input_tensor.device)
    scales = torch.empty(num_rows, dtype=torch.float32, device=input_tensor.device)

    row_size_padded = triton.next_power_of_2(row_size)

    grid = (num_rows,)
    _fused_quantize_encode_hamming74_kernel[grid](
        flat_input,
        codewords,
        scales,
        num_rows,
        row_size,
        ROW_SIZE_PADDED=row_size_padded,
    )

    if input_tensor.dim() == 1:
        codewords = codewords.squeeze(0)
    else:
        codewords = codewords.view(original_shape)
        scales = scales.view(original_shape[:-1])

    return codewords, scales


@triton.jit
def _fused_decode_dequantize_hamming84_kernel(
    codeword_ptr,    # [num_rows, row_size] uint8 codewords
    output_ptr,      # [num_rows, row_size] float32/float16 output
    scale_ptr,       # [num_rows] float32 scales
    error_count_ptr, # [1] total corrected errors (atomic)
    lut_ptr,         # [8] syndrome LUT
    num_rows,
    row_size,
    ROW_SIZE_PADDED: tl.constexpr,
):
    """
    Fused kernel: Hamming84 decode + INT4 dequantize.
    """
    pid = tl.program_id(0)

    if pid >= num_rows:
        return

    row_input_ptr = codeword_ptr + pid * row_size
    row_output_ptr = output_ptr + pid * row_size

    offsets = tl.arange(0, ROW_SIZE_PADDED)
    mask = offsets < row_size

    # Load codewords
    codewords = tl.load(row_input_ptr + offsets, mask=mask, other=0).to(tl.uint8)

    # Load scale for this row
    scale = tl.load(scale_ptr + pid)

    # Hamming84 decode inline
    hamming7 = codewords & 0x7F
    stored_parity = (codewords >> 7) & 1

    c0 = (hamming7 >> 0) & 1
    c1 = (hamming7 >> 1) & 1
    c2 = (hamming7 >> 2) & 1
    c3 = (hamming7 >> 3) & 1
    c4 = (hamming7 >> 4) & 1
    c5 = (hamming7 >> 5) & 1
    c6 = (hamming7 >> 6) & 1

    s0 = c0 ^ c1 ^ c3 ^ c4
    s1 = c0 ^ c2 ^ c3 ^ c5
    s2 = c1 ^ c2 ^ c3 ^ c6

    syndrome = (s0 | (s1 << 1) | (s2 << 2)).to(tl.int32)

    # Compute actual parity
    actual_parity = hamming7 ^ (hamming7 >> 4)
    actual_parity = actual_parity ^ (actual_parity >> 2)
    actual_parity = actual_parity ^ (actual_parity >> 1)
    actual_parity = actual_parity & 1

    parity_error = stored_parity != actual_parity

    # Determine error type: 0=none, 1=single, 2=double, 3=parity_only
    syndrome_zero = syndrome == 0
    error_type = tl.where(
        syndrome_zero, tl.where(parity_error, 3, 0), tl.where(parity_error, 1, 2)
    ).to(tl.uint8)

    # LUT lookup for error position
    error_pos = tl.load(lut_ptr + syndrome, mask=mask, other=-1)

    # Correct single-bit errors
    should_correct = (error_type == 1) & (error_pos >= 0)
    correction_mask = tl.where(should_correct, 1 << error_pos, 0).to(tl.uint8)
    corrected = hamming7 ^ correction_mask

    # For double errors, output 0 (could also flag)
    corrected = tl.where(error_type == 2, 0, corrected).to(tl.uint8)

    # Extract 4-bit data
    int4_val = corrected & 0x0F

    # Count errors corrected
    errors_in_row = tl.sum((error_type == 1).to(tl.int32))
    tl.atomic_add(error_count_ptr, errors_in_row)

    # Dequantize: (int4 - 8) * scale
    dequantized = (int4_val.to(tl.float32) - 8.0) * scale

    # Store output
    tl.store(row_output_ptr + offsets, dequantized, mask=mask)


# Cache for syndrome LUT
_decode_lut_cache = {}


def _get_decode_lut_gpu(device):
    """Get syndrome LUT on GPU for decode (cached)."""
    from .config import SYNDROME_LUT_HAMMING84
    if device not in _decode_lut_cache:
        _decode_lut_cache[device] = SYNDROME_LUT_HAMMING84.to(device)
    return _decode_lut_cache[device]


def fused_decode_dequantize_hamming84(
    codewords: torch.Tensor,
    scales: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """
    Fused Hamming84 decode and dequantization in a single kernel launch.

    Args:
        codewords: uint8 tensor of shape [..., row_size]
        scales: float32 tensor of shape [...]  (matches codewords without last dim)
        output_dtype: Output tensor dtype (float32 or float16)

    Returns:
        tuple: (dequantized_output, errors_corrected)
    """
    assert codewords.is_cuda, "Input must be on CUDA device"
    assert scales.is_cuda, "Scales must be on CUDA device"

    original_shape = codewords.shape
    row_size = original_shape[-1]

    if codewords.dim() == 1:
        flat_codewords = codewords.unsqueeze(0)
        flat_scales = scales.unsqueeze(0)
        num_rows = 1
    else:
        flat_codewords = codewords.reshape(-1, row_size)
        flat_scales = scales.flatten()
        num_rows = flat_codewords.shape[0]

    flat_codewords = flat_codewords.contiguous()

    # Allocate output
    output = torch.empty(num_rows, row_size, dtype=torch.float32, device=codewords.device)
    error_count = torch.zeros(1, dtype=torch.int32, device=codewords.device)

    # Get LUT
    lut_gpu = _get_decode_lut_gpu(codewords.device)

    row_size_padded = triton.next_power_of_2(row_size)

    grid = (num_rows,)
    _fused_decode_dequantize_hamming84_kernel[grid](
        flat_codewords,
        output,
        flat_scales,
        error_count,
        lut_gpu,
        num_rows,
        row_size,
        ROW_SIZE_PADDED=row_size_padded,
    )

    errors_corrected = int(error_count.item())

    # Reshape and convert dtype
    if codewords.dim() == 1:
        output = output.squeeze(0)
    else:
        output = output.view(original_shape)

    if output_dtype != torch.float32:
        output = output.to(output_dtype)

    return output, errors_corrected
