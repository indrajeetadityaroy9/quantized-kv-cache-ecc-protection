import pytest
import torch


class TestMemoryLayout:
    def test_ecc_cache_config_hamming84(self):
        from vllm_kernels.memory_layout import ECCCacheConfig

        config = ECCCacheConfig(
            num_heads=32,
            head_size=128,
            num_layers=32,
            block_size=16,
            num_blocks=256,
            codec="hamming84",
        )

        assert config.dtype == torch.uint8
        assert config.values_per_block == 16 * 128
        assert config.codewords_per_block == 16 * 128
        assert config.storage_overhead == 2.0

    def test_ecc_cache_config_golay(self):
        from vllm_kernels.memory_layout import ECCCacheConfig

        config = ECCCacheConfig(
            num_heads=32,
            head_size=128,
            num_layers=32,
            block_size=16,
            num_blocks=256,
            codec="golay",
        )

        assert config.dtype == torch.int32

        expected_codewords = (16 * 128 + 2) // 3
        assert config.codewords_per_block == expected_codewords
        assert config.storage_overhead == pytest.approx(32 / 12, rel=0.01)

    def test_allocate_ecc_kv_cache(self):
        from vllm_kernels.memory_layout import ECCCacheConfig, allocate_ecc_kv_cache

        config = ECCCacheConfig(
            num_heads=8,
            head_size=64,
            num_layers=4,
            block_size=16,
            num_blocks=32,
            codec="hamming84",
        )

        key_cache, value_cache = allocate_ecc_kv_cache(config, device="cuda")

        expected_shape = (32, 4, 8, config.codewords_per_block)
        assert key_cache.shape == expected_shape
        assert value_cache.shape == expected_shape
        assert key_cache.dtype == torch.uint8
        assert key_cache.device.type == "cuda"

    def test_create_block_table(self):
        from vllm_kernels.memory_layout import create_block_table

        batch_size = 4
        max_seq_len = 512
        block_size = 16

        block_table = create_block_table(
            batch_size, max_seq_len, block_size, device="cuda"
        )

        expected_max_blocks = (max_seq_len + block_size - 1) // block_size
        assert block_table.shape == (batch_size, expected_max_blocks)
        assert block_table.dtype == torch.int32
        assert (block_table == -1).all()

    def test_allocate_blocks(self):
        from vllm_kernels.memory_layout import create_block_table, allocate_blocks

        batch_size = 2
        max_seq_len = 256
        block_size = 16
        num_blocks = 64

        block_table = create_block_table(
            batch_size, max_seq_len, block_size, device="cuda"
        )
        free_blocks = torch.arange(num_blocks, device="cuda")

        next_free = allocate_blocks(
            block_table,
            batch_idx=0,
            num_blocks_needed=5,
            free_blocks=free_blocks,
            next_free_idx=0,
        )

        assert next_free == 5
        assert (block_table[0, :5] == torch.arange(5, device="cuda")).all()
        assert (block_table[0, 5:] == -1).all()

    def test_compute_slot_mapping(self):
        from vllm_kernels.memory_layout import (
            create_block_table,
            allocate_blocks,
            compute_slot_mapping,
        )

        block_size = 16
        seq_len = 50

        block_table = create_block_table(1, 256, block_size, device="cuda")
        free_blocks = torch.arange(100, device="cuda")
        allocate_blocks(block_table, 0, 4, free_blocks, 0)

        slot_mapping = compute_slot_mapping(
            seq_len, block_size, block_table, batch_idx=0
        )

        assert slot_mapping.shape == (seq_len, 2)

        assert slot_mapping[0, 0].item() == 0
        assert slot_mapping[0, 1].item() == 0

        assert slot_mapping[16, 0].item() == 1
        assert slot_mapping[16, 1].item() == 0

    def test_get_codec_for_block(self):
        from vllm_kernels.memory_layout import get_codec_for_block

        for i in range(4):
            assert get_codec_for_block(i, sink_blocks=4) == "golay"

        for i in range(4, 10):
            assert get_codec_for_block(i, sink_blocks=4) == "hamming84"


class TestCacheWrite:
    def test_compute_quantization_scales(self):
        from vllm_kernels.paged_cache_ecc import compute_quantization_scales

        tensor = torch.tensor(
            [
                [1.0, -2.0, 3.0, -4.0],
                [0.5, -0.5, 0.1, -0.1],
            ],
            device="cuda",
        )

        scales = compute_quantization_scales(tensor, dim=-1)

        assert scales.shape == (2,)

        assert scales[0].item() == pytest.approx(4.0 / 7, rel=0.01)

        assert scales[1].item() == pytest.approx(0.5 / 7, rel=0.01)

    def test_write_kv_simple_hamming84(self):
        from vllm_kernels.paged_cache_ecc import write_kv_to_cache_simple

        batch_size = 2
        seq_len = 32
        hidden = 64

        kv = torch.randn(
            batch_size, seq_len, hidden, device="cuda", dtype=torch.float16
        )

        encoded, scales = write_kv_to_cache_simple(kv, codec="hamming84")

        assert encoded.shape == kv.shape
        assert encoded.dtype == torch.uint8
        assert scales.shape == (batch_size, seq_len)

    def test_write_kv_roundtrip_hamming84(self):
        from vllm_kernels.paged_cache_ecc import (
            write_kv_to_cache_simple,
            compute_quantization_scales,
        )
        from hamming74.triton_kernels import hamming84_decode

        batch_size = 2
        seq_len = 16
        hidden = 32

        kv = torch.randn(
            batch_size, seq_len, hidden, device="cuda", dtype=torch.float16
        )
        scales = compute_quantization_scales(kv.float(), dim=-1)

        encoded, _ = write_kv_to_cache_simple(kv, codec="hamming84", scale=scales)

        decoded, stats = hamming84_decode(encoded.flatten())
        decoded = decoded.view(encoded.shape)

        dequantized = (decoded.float() - 8) * scales.unsqueeze(-1).cuda()

        mse = ((kv.float() - dequantized) ** 2).mean()
        assert mse < 1.0, f"MSE too high: {mse}"


class TestBenchmarkHarness:
    def test_cuda_timer(self):
        from vllm_kernels.benchmark_harness import cuda_timer

        data = torch.randn(1000, device="cuda")

        latency = cuda_timer(lambda: data + 1, warmup=5, repeat=50)

        assert latency > 0
        assert latency < 1000

    def test_benchmark_hamming84_encode(self):
        from vllm_kernels.benchmark_harness import benchmark_hamming84_encode

        result = benchmark_hamming84_encode(n_elements=10_000, warmup=5, repeat=20)

        assert result.name == "hamming84_encode"
        assert result.n_elements == 10_000
        assert result.latency_us > 0
        assert result.throughput_mvals_sec > 0

    def test_benchmark_golay_encode(self):
        from vllm_kernels.benchmark_harness import benchmark_golay_encode

        result = benchmark_golay_encode(n_triplets=3_333, warmup=5, repeat=20)

        assert result.name == "golay_encode"
        assert result.latency_us > 0
        assert result.throughput_mvals_sec > 0

    def test_benchmark_fault_injection(self):
        from vllm_kernels.benchmark_harness import benchmark_fault_injection

        result = benchmark_fault_injection(
            n_elements=10_000, ber=0.05, warmup=5, repeat=20
        )

        assert "fault_injection" in result.name
        assert result.extra["ber"] == 0.05

    def test_benchmark_encode_inject_decode_pipeline(self):
        from vllm_kernels.benchmark_harness import benchmark_encode_inject_decode

        result = benchmark_encode_inject_decode(
            codec="hamming84", n_elements=10_000, ber=0.01, warmup=5, repeat=20
        )

        assert "pipeline" in result.name
        assert result.extra["codec"] == "hamming84"


class TestIntegration:
    def test_full_ecc_pipeline_hamming84(self):
        from hamming74.triton_kernels import (
            hamming84_encode,
            hamming84_decode,
            inject_bit_errors_triton,
        )

        n_values = 10_000
        original = torch.randint(0, 16, (n_values,), dtype=torch.uint8, device="cuda")

        encoded = hamming84_encode(original)

        corrupted, stats = inject_bit_errors_triton(
            encoded, ber=0.001, n_bits=8, seed=42, return_stats=True
        )

        decoded, decode_stats = hamming84_decode(corrupted)

        accuracy = (decoded == original).float().mean()
        assert accuracy > 0.99, f"Accuracy too low: {accuracy}"

    def test_full_ecc_pipeline_golay(self):
        from hamming74.triton_kernels import (
            golay_encode,
            golay_decode,
            inject_bit_errors_triton,
        )

        n_triplets = 3_333
        original = torch.randint(
            0, 16, (n_triplets, 3), dtype=torch.uint8, device="cuda"
        )

        encoded = golay_encode(original)

        corrupted, stats = inject_bit_errors_triton(
            encoded, ber=0.01, n_bits=24, seed=42, return_stats=True
        )

        decoded, decode_stats = golay_decode(corrupted)

        accuracy = (decoded == original).float().mean()
        assert accuracy > 0.98, f"Accuracy too low: {accuracy}"
