import pytest
import torch


class TestTritonInterpolation:
    def test_no_double_errors_unchanged(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([1, 5, 10, 15, 8], dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)

        result = interpolate_double_errors(q, err)

        assert torch.equal(result, q), "No-error case should return exact copy"

    def test_single_double_error_middle(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([4, 8, 12, 8, 4], dtype=torch.uint8, device="cuda")
        err = torch.tensor(
            [0, 0, ErrorType.DOUBLE_DETECTED, 0, 0], dtype=torch.uint8, device="cuda"
        )

        result = interpolate_double_errors(q, err)

        expected = torch.tensor([4, 8, 8, 8, 4], dtype=torch.uint8, device="cuda")
        assert torch.equal(
            result, expected
        ), f"Middle interpolation failed: {result} vs {expected}"

    def test_double_error_first_position(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([15, 4, 8, 12], dtype=torch.uint8, device="cuda")
        err = torch.tensor(
            [ErrorType.DOUBLE_DETECTED, 0, 0, 0], dtype=torch.uint8, device="cuda"
        )

        result = interpolate_double_errors(q, err)

        expected = torch.tensor([10, 4, 8, 12], dtype=torch.uint8, device="cuda")
        assert torch.equal(
            result, expected
        ), f"Left boundary failed: {result} vs {expected}"

    def test_double_error_last_position(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([4, 8, 12, 15], dtype=torch.uint8, device="cuda")
        err = torch.tensor(
            [0, 0, 0, ErrorType.DOUBLE_DETECTED], dtype=torch.uint8, device="cuda"
        )

        result = interpolate_double_errors(q, err)

        expected = torch.tensor([4, 8, 12, 14], dtype=torch.uint8, device="cuda")
        assert torch.equal(
            result, expected
        ), f"Right boundary failed: {result} vs {expected}"

    def test_multiple_scattered_double_errors(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([0, 4, 8, 12, 8, 4, 0], dtype=torch.uint8, device="cuda")
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
            device="cuda",
        )

        result = interpolate_double_errors(q, err)

        expected = torch.tensor([0, 4, 8, 8, 8, 4, 0], dtype=torch.uint8, device="cuda")
        assert torch.equal(
            result, expected
        ), f"Scattered errors failed: {result} vs {expected}"

    def test_consecutive_double_errors(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor([4, 0, 0, 0, 4], dtype=torch.uint8, device="cuda")
        err = torch.tensor(
            [
                0,
                ErrorType.DOUBLE_DETECTED,
                ErrorType.DOUBLE_DETECTED,
                ErrorType.DOUBLE_DETECTED,
                0,
            ],
            dtype=torch.uint8,
            device="cuda",
        )

        result = interpolate_double_errors(q, err)

        expected = torch.tensor([4, 2, 0, 2, 4], dtype=torch.uint8, device="cuda")
        assert torch.equal(
            result, expected
        ), f"Consecutive errors failed: {result} vs {expected}"

    def test_large_tensor_non_double_unchanged(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        N = 100000
        q = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda")
        err = torch.zeros(N, dtype=torch.uint8, device="cuda")

        double_mask = torch.rand(N, device="cuda") < 0.1
        err[double_mask] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        non_double = ~double_mask
        assert torch.equal(
            result[non_double], q[non_double]
        ), "Non-double positions should be unchanged"

    def test_output_in_valid_range(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        N = 100000
        q = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda")
        err = torch.zeros(N, dtype=torch.uint8, device="cuda")

        err[torch.rand(N, device="cuda") < 0.2] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        assert (result >= 0).all(), "Result has values below 0"
        assert (result <= 15).all(), "Result has values above 15"


class TestTritonInterpolation2D:
    def test_2d_shape_preserved(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (32, 1024), dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)
        err[::4, ::10] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        assert result.shape == q.shape, "Shape should be preserved"

    def test_2d_non_double_unchanged(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (32, 1024), dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)
        err[::4, ::10] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        non_double = err != ErrorType.DOUBLE_DETECTED
        assert torch.equal(
            result[non_double], q[non_double]
        ), "Non-double positions should be unchanged (2D)"

    def test_2d_batch_independent(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.tensor(
            [
                [4, 0, 8, 4],
                [4, 8, 0, 4],
            ],
            dtype=torch.uint8,
            device="cuda",
        )

        err = torch.tensor(
            [
                [0, ErrorType.DOUBLE_DETECTED, 0, 0],
                [0, 0, ErrorType.DOUBLE_DETECTED, 0],
            ],
            dtype=torch.uint8,
            device="cuda",
        )

        result = interpolate_double_errors(q, err)

        expected = torch.tensor(
            [
                [4, 6, 8, 4],
                [4, 8, 6, 4],
            ],
            dtype=torch.uint8,
            device="cuda",
        )

        assert torch.equal(result, expected), f"2D batch failed: {result} vs {expected}"


class TestTritonInterpolationPerformance:
    def test_handles_empty_tensor(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )

        empty_q = torch.tensor([], dtype=torch.uint8, device="cuda")
        empty_err = torch.tensor([], dtype=torch.uint8, device="cuda")

        result = interpolate_double_errors(empty_q, empty_err)

        assert result.numel() == 0

    def test_handles_single_element(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        single_q = torch.tensor([8], dtype=torch.uint8, device="cuda")
        single_err = torch.tensor(
            [ErrorType.DOUBLE_DETECTED], dtype=torch.uint8, device="cuda"
        )

        result = interpolate_double_errors(single_q, single_err)

        assert result.item() == 8

    def test_early_exit_no_double_errors(self):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (1000000,), dtype=torch.uint8, device="cuda")
        err = torch.zeros_like(q)

        result = interpolate_double_errors(q, err)

        assert torch.equal(result, q)

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000])
    def test_various_sizes(self, size):
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        q = torch.randint(0, 16, (size,), dtype=torch.uint8, device="cuda")
        err = torch.zeros(size, dtype=torch.uint8, device="cuda")

        if size > 1:
            err[::3] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(q, err)

        assert result.shape == q.shape
        assert (result >= 0).all() and (result <= 15).all()


class TestTritonInterpolationWithHamming84:
    def test_end_to_end_with_double_errors(self):
        from hamming74.triton_kernels.hamming84_triton import (
            hamming84_encode,
            hamming84_decode,
        )
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        original = torch.tensor([5, 10, 5], dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(original)

        corrupted = encoded.clone()
        corrupted[1] ^= 0b11

        decoded, error_types, _ = hamming84_decode(corrupted, return_error_types=True)

        assert (
            error_types[1].item() == ErrorType.DOUBLE_DETECTED
        ), f"Expected DOUBLE_DETECTED, got {error_types[1].item()}"

        result = interpolate_double_errors(decoded, error_types)

        assert result[0].item() == 5
        assert result[1].item() == 5
        assert result[2].item() == 5

    def test_large_scale_with_injected_double_errors(self):
        from hamming74.triton_kernels.hamming84_triton import (
            hamming84_encode,
            hamming84_decode,
        )
        from hamming74.triton_kernels.interpolation_triton import (
            interpolate_double_errors,
        )
        from hamming74.triton_kernels.config import ErrorType

        N = 10000
        original = torch.randint(0, 16, (N,), dtype=torch.uint8, device="cuda")
        encoded = hamming84_encode(original)

        corrupted = encoded.clone()
        double_error_mask = torch.rand(N, device="cuda") < 0.1
        double_error_positions = torch.where(double_error_mask)[0]

        for pos in double_error_positions:
            bit1 = torch.randint(0, 7, (1,)).item()
            bit2 = (bit1 + torch.randint(1, 7, (1,)).item()) % 7
            corrupted[pos] ^= (1 << bit1) ^ (1 << bit2)

        decoded, error_types, _ = hamming84_decode(corrupted, return_error_types=True)
        result = interpolate_double_errors(decoded, error_types)

        assert (result >= 0).all() and (result <= 15).all()

        clean_positions = ~double_error_mask

        matches = (result[clean_positions] == original[clean_positions]).float().mean()
        assert matches > 0.9, f"Only {matches*100:.1f}% of clean positions match"
