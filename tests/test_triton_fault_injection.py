import pytest
import torch


class TestTritonFaultInjectionBERFidelity:
    def test_zero_ber_no_errors(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.randint(0, 256, (10000,), dtype=torch.uint8, device="cuda")
        corrupted, stats = inject_bit_errors_triton(
            data, ber=0.0, n_bits=8, seed=42, return_stats=True
        )

        assert torch.equal(data, corrupted), "Zero BER should not change data"
        assert stats[0] == 0, "Should have 0 errors"

    @pytest.mark.parametrize("target_ber", [0.01, 0.05, 0.10, 0.20])
    def test_ber_fidelity_uint8(self, target_ber):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        n_values = 100_000
        n_bits = 8
        data = torch.zeros(n_values, dtype=torch.uint8, device="cuda")

        corrupted, stats = inject_bit_errors_triton(
            data, ber=target_ber, n_bits=n_bits, seed=42, return_stats=True
        )

        total_bits = n_values * n_bits
        empirical_ber = stats[0] / total_bits
        deviation = abs(empirical_ber - target_ber)

        tolerance = max(target_ber * 0.1, 0.01)
        assert (
            deviation < tolerance
        ), f"BER={target_ber}: empirical={empirical_ber:.4f}, deviation={deviation:.4f}"

    @pytest.mark.parametrize("target_ber", [0.01, 0.05, 0.10])
    def test_ber_fidelity_int32_24bit(self, target_ber):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        n_values = 50_000
        n_bits = 24
        data = torch.zeros(n_values, dtype=torch.int32, device="cuda")

        corrupted, stats = inject_bit_errors_triton(
            data, ber=target_ber, n_bits=n_bits, seed=42, return_stats=True
        )

        total_bits = n_values * n_bits
        empirical_ber = stats[0] / total_bits
        deviation = abs(empirical_ber - target_ber)

        tolerance = max(target_ber * 0.1, 0.01)
        assert (
            deviation < tolerance
        ), f"BER={target_ber}: empirical={empirical_ber:.4f}, deviation={deviation:.4f}"


class TestTritonFaultInjectionDeterminism:
    def test_same_seed_same_result_uint8(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.randint(0, 256, (10000,), dtype=torch.uint8, device="cuda")

        corrupted1 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)
        corrupted2 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)

        assert torch.equal(
            corrupted1, corrupted2
        ), "Same seed should produce same errors"

    def test_same_seed_same_result_int32(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.randint(0, 2**20, (10000,), dtype=torch.int32, device="cuda")

        corrupted1 = inject_bit_errors_triton(data, ber=0.1, n_bits=24, seed=42)
        corrupted2 = inject_bit_errors_triton(data, ber=0.1, n_bits=24, seed=42)

        assert torch.equal(
            corrupted1, corrupted2
        ), "Same seed should produce same errors"

    def test_different_seed_different_result(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")

        corrupted1 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)
        corrupted2 = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=99)

        assert not torch.equal(
            corrupted1, corrupted2
        ), "Different seeds should produce different errors"


class TestTritonFaultInjectionCorrectness:
    def test_only_active_bits_affected(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        n_bits = 4

        data = torch.full((10000,), 0x80, dtype=torch.uint8, device="cuda")

        corrupted = inject_bit_errors_triton(data, ber=0.5, n_bits=n_bits, seed=42)

        high_bits_unchanged = ((corrupted >> 7) & 1) == 1
        assert high_bits_unchanged.all(), "Bits outside n_bits should not be affected"

    def test_xor_relationship(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.arange(256, dtype=torch.uint8, device="cuda")
        corrupted = inject_bit_errors_triton(data.clone(), ber=0.1, n_bits=8, seed=42)

        error_mask = data ^ corrupted

        recovered = corrupted ^ error_mask
        assert torch.equal(recovered, data), "XOR relationship should hold"


class TestTritonFaultInjectionVariousSizes:
    @pytest.mark.parametrize("size", [1, 100, 1024, 10000, 100000])
    def test_various_sizes_uint8(self, size):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.randint(0, 256, (size,), dtype=torch.uint8, device="cuda")
        corrupted = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)

        assert corrupted.shape == data.shape
        assert corrupted.dtype == data.dtype

    @pytest.mark.parametrize("size", [1, 100, 1024, 10000])
    def test_various_sizes_int32(self, size):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.randint(0, 2**20, (size,), dtype=torch.int32, device="cuda")
        corrupted = inject_bit_errors_triton(data, ber=0.1, n_bits=24, seed=42)

        assert corrupted.shape == data.shape
        assert corrupted.dtype == data.dtype

    def test_empty_tensor(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.empty(0, dtype=torch.uint8, device="cuda")
        corrupted = inject_bit_errors_triton(data, ber=0.1, n_bits=8, seed=42)

        assert corrupted.numel() == 0


class TestTritonFaultInjectionStatistics:
    def test_stats_match_actual_errors(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")
        corrupted, stats = inject_bit_errors_triton(
            data, ber=0.1, n_bits=8, seed=42, return_stats=True
        )

        actual_errors = 0
        for bit in range(8):
            actual_errors += int(((corrupted >> bit) & 1).sum())

        assert (
            stats[0] == actual_errors
        ), f"Reported errors {stats[0]} != actual errors {actual_errors}"

    def test_elements_with_errors_count(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton,
        )

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")
        corrupted, stats = inject_bit_errors_triton(
            data, ber=0.1, n_bits=8, seed=42, return_stats=True
        )

        actual_elements_with_errors = int((corrupted != data).sum())

        assert (
            stats[1] == actual_elements_with_errors
        ), f"Reported elements with errors {stats[1]} != actual {actual_elements_with_errors}"


class TestTritonFaultInjectionBatchedAPI:
    def test_batched_api_returns_error_count(self):
        from ecc_codecs.triton_kernels.fault_injection_triton import (
            inject_bit_errors_triton_batched,
        )

        data = torch.zeros(10000, dtype=torch.uint8, device="cuda")
        corrupted, total_errors = inject_bit_errors_triton_batched(
            data, ber=0.1, n_bits=8, seed=42
        )

        assert corrupted.shape == data.shape
        assert isinstance(total_errors, int)
        assert total_errors > 0
