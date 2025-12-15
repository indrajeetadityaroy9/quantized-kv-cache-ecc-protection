import torch
import triton
import triton.language as tl

from .config import INTERPOLATION_BLOCK_SIZE, ErrorType


@triton.jit
def interpolate_double_errors_kernel(
    q_ptr,
    error_type_ptr,
    output_ptr,
    seq_len,
    num_sequences,
    BLOCK_SIZE: tl.constexpr,
    DOUBLE_DETECTED: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_per_seq = tl.cdiv(seq_len, BLOCK_SIZE)
    seq_idx = pid // num_blocks_per_seq
    block_idx = pid % num_blocks_per_seq

    local_offset = block_idx * BLOCK_SIZE
    offsets = local_offset + tl.arange(0, BLOCK_SIZE)
    mask = (seq_idx < num_sequences) & (offsets < seq_len)

    base = seq_idx * seq_len

    q = tl.load(q_ptr + base + offsets, mask=mask, other=0).to(tl.float32)
    err = tl.load(error_type_ptr + base + offsets, mask=mask, other=0)

    is_double = err == DOUBLE_DETECTED

    left_idx = tl.maximum(offsets - 1, 0)
    right_idx = tl.minimum(offsets + 1, seq_len - 1)

    left_val = tl.load(q_ptr + base + left_idx, mask=mask, other=0).to(tl.float32)
    right_val = tl.load(q_ptr + base + right_idx, mask=mask, other=0).to(tl.float32)

    interpolated = (left_val + right_val) * 0.5

    result = tl.where(is_double, interpolated, q)

    result = tl.maximum(0.0, tl.minimum(15.0, result + 0.5))
    result = result.to(tl.uint8)

    tl.store(output_ptr + base + offsets, result, mask=mask)


def interpolate_double_errors(q, error_type, original_shape=None, seq_dim=-1):
    assert q.is_cuda, "Input must be on CUDA device"
    assert error_type.is_cuda, "Error type must be on CUDA device"
    assert q.shape == error_type.shape, "Shape mismatch between q and error_type"

    has_double_errors = (error_type == ErrorType.DOUBLE_DETECTED).any()
    if not has_double_errors:
        return q.clone()

    input_shape = q.shape

    if q.dim() == 1:
        flat_q = q.unsqueeze(0)
        flat_err = error_type.unsqueeze(0)
    elif q.dim() == 2:
        flat_q = q
        flat_err = error_type
    else:
        if seq_dim < 0:
            seq_dim = q.dim() + seq_dim

        if seq_dim != q.dim() - 1:
            perm = list(range(q.dim()))
            perm.remove(seq_dim)
            perm.append(seq_dim)
            q = q.permute(*perm)
            error_type = error_type.permute(*perm)

        seq_len = q.shape[-1]
        batch_size = q.numel() // seq_len
        flat_q = q.reshape(batch_size, seq_len)
        flat_err = error_type.reshape(batch_size, seq_len)

    flat_q = flat_q.contiguous().to(torch.uint8)
    flat_err = flat_err.contiguous().to(torch.uint8)

    num_sequences, seq_len = flat_q.shape

    output = torch.empty_like(flat_q)

    num_blocks_per_seq = triton.cdiv(seq_len, INTERPOLATION_BLOCK_SIZE)
    total_blocks = num_sequences * num_blocks_per_seq

    interpolate_double_errors_kernel[(total_blocks,)](
        flat_q,
        flat_err,
        output,
        seq_len,
        num_sequences,
        BLOCK_SIZE=INTERPOLATION_BLOCK_SIZE,
        DOUBLE_DETECTED=ErrorType.DOUBLE_DETECTED,
    )

    return output.reshape(input_shape)


def interpolate_double_errors_1d(q, error_type):
    return interpolate_double_errors(q, error_type, seq_dim=-1)


def verify_triton_vs_cpu():
    device = "cuda"

    print("Verifying Triton interpolation vs CPU reference...")

    q = torch.tensor([1, 5, 10, 15, 8], dtype=torch.uint8, device=device)
    err = torch.zeros_like(q)
    result = interpolate_double_errors(q, err)
    assert torch.equal(result, q), "No-error case should return exact copy"
    print("  [PASS] No double errors -> exact copy")

    q = torch.tensor([4, 8, 12, 8, 4], dtype=torch.uint8, device=device)
    err = torch.tensor(
        [0, 0, ErrorType.DOUBLE_DETECTED, 0, 0], dtype=torch.uint8, device=device
    )
    result = interpolate_double_errors(q, err)

    expected = torch.tensor([4, 8, 8, 8, 4], dtype=torch.uint8, device=device)
    assert torch.equal(
        result, expected
    ), f"Middle interpolation failed: {result} vs {expected}"
    print("  [PASS] Single double error in middle -> neighbor average")

    q = torch.tensor([15, 4, 8, 12], dtype=torch.uint8, device=device)
    err = torch.tensor(
        [ErrorType.DOUBLE_DETECTED, 0, 0, 0], dtype=torch.uint8, device=device
    )
    result = interpolate_double_errors(q, err)

    expected = torch.tensor([10, 4, 8, 12], dtype=torch.uint8, device=device)
    assert torch.equal(
        result, expected
    ), f"Left boundary failed: {result} vs {expected}"
    print("  [PASS] Double error at first position -> boundary handling")

    q = torch.tensor([4, 8, 12, 15], dtype=torch.uint8, device=device)
    err = torch.tensor(
        [0, 0, 0, ErrorType.DOUBLE_DETECTED], dtype=torch.uint8, device=device
    )
    result = interpolate_double_errors(q, err)

    expected = torch.tensor([4, 8, 12, 14], dtype=torch.uint8, device=device)
    assert torch.equal(
        result, expected
    ), f"Right boundary failed: {result} vs {expected}"
    print("  [PASS] Double error at last position -> boundary handling")

    q = torch.tensor([0, 4, 8, 12, 8, 4, 0], dtype=torch.uint8, device=device)
    err = torch.tensor(
        [
            0,
            ErrorType.DOUBLE_DETECTED,
            0,
            ErrorType.DOUBLE_DETECTED,
            0,
            ErrorType.DOUBLE_DETECTED,
            0,
        ],
        dtype=torch.uint8,
        device=device,
    )
    result = interpolate_double_errors(q, err)

    expected = torch.tensor([0, 4, 8, 8, 8, 4, 0], dtype=torch.uint8, device=device)
    assert torch.equal(
        result, expected
    ), f"Scattered errors failed: {result} vs {expected}"
    print("  [PASS] Multiple scattered double errors")

    N = 100000
    q = torch.randint(0, 16, (N,), dtype=torch.uint8, device=device)
    err = torch.zeros(N, dtype=torch.uint8, device=device)

    double_mask = torch.rand(N, device=device) < 0.1
    err[double_mask] = ErrorType.DOUBLE_DETECTED

    result = interpolate_double_errors(q, err)

    non_double = ~double_mask
    assert torch.equal(
        result[non_double], q[non_double]
    ), "Non-double positions should be unchanged"

    assert (result >= 0).all() and (
        result <= 15
    ).all(), "Results should be in valid INT4 range"

    num_double = double_mask.sum().item()
    print(
        f"  [PASS] Large tensor with {num_double} double errors ({100*num_double/N:.1f}%)"
    )

    q_2d = torch.randint(0, 16, (32, 1024), dtype=torch.uint8, device=device)
    err_2d = torch.zeros_like(q_2d)

    err_2d[::4, ::10] = ErrorType.DOUBLE_DETECTED

    result_2d = interpolate_double_errors(q_2d, err_2d)
    assert result_2d.shape == q_2d.shape, "Shape should be preserved"

    non_double_2d = err_2d != ErrorType.DOUBLE_DETECTED
    assert torch.equal(
        result_2d[non_double_2d], q_2d[non_double_2d]
    ), "Non-double positions should be unchanged (2D)"
    print("  [PASS] 2D tensor (batch of sequences)")

    print("All Triton interpolation verifications passed!")
    return True


if __name__ == "__main__":
    verify_triton_vs_cpu()
