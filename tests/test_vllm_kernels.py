"""
Tests for vLLM PagedAttention ECC integration kernels.
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestMemoryLayout:
    """Tests for memory layout utilities."""

    def test_ecc_cache_config_hamming84(self):
        """Test ECCCacheConfig for Hamming84."""
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
        assert config.values_per_block == 16 * 128  # block_size * head_size
        assert config.codewords_per_block == 16 * 128  # 1:1 for Hamming84
        assert config.storage_overhead == 2.0  # 8 bits vs 4 bits

    def test_ecc_cache_config_golay(self):
        """Test ECCCacheConfig for Golay."""
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
        # Golay packs 3 values per codeword
        expected_codewords = (16 * 128 + 2) // 3
        assert config.codewords_per_block == expected_codewords
        assert config.storage_overhead == pytest.approx(32 / 12, rel=0.01)

    def test_allocate_ecc_kv_cache(self):
        """Test ECC cache allocation."""
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
        """Test block table creation."""
        from vllm_kernels.memory_layout import create_block_table

        batch_size = 4
        max_seq_len = 512
        block_size = 16

        block_table = create_block_table(batch_size, max_seq_len, block_size, device="cuda")

        expected_max_blocks = (max_seq_len + block_size - 1) // block_size
        assert block_table.shape == (batch_size, expected_max_blocks)
        assert block_table.dtype == torch.int32
        assert (block_table == -1).all()  # Initially unallocated

    def test_allocate_blocks(self):
        """Test block allocation."""
        from vllm_kernels.memory_layout import create_block_table, allocate_blocks

        batch_size = 2
        max_seq_len = 256
        block_size = 16
        num_blocks = 64

        block_table = create_block_table(batch_size, max_seq_len, block_size, device="cuda")
        free_blocks = torch.arange(num_blocks, device="cuda")

        # Allocate 5 blocks for batch 0
        next_free = allocate_blocks(block_table, batch_idx=0, num_blocks_needed=5,
                                     free_blocks=free_blocks, next_free_idx=0)

        assert next_free == 5
        assert (block_table[0, :5] == torch.arange(5, device="cuda")).all()
        assert (block_table[0, 5:] == -1).all()

    def test_compute_slot_mapping(self):
        """Test slot mapping computation."""
        from vllm_kernels.memory_layout import (
            create_block_table, allocate_blocks, compute_slot_mapping
        )

        block_size = 16
        seq_len = 50

        block_table = create_block_table(1, 256, block_size, device="cuda")
        free_blocks = torch.arange(100, device="cuda")
        allocate_blocks(block_table, 0, 4, free_blocks, 0)

        slot_mapping = compute_slot_mapping(seq_len, block_size, block_table, batch_idx=0)

        assert slot_mapping.shape == (seq_len, 2)

        # Check first token (should be in block 0, slot 0)
        assert slot_mapping[0, 0].item() == 0  # physical block
        assert slot_mapping[0, 1].item() == 0  # slot

        # Check token at position 16 (should be in block 1, slot 0)
        assert slot_mapping[16, 0].item() == 1
        assert slot_mapping[16, 1].item() == 0

    def test_get_codec_for_block(self):
        """Test adaptive UEP codec selection."""
        from vllm_kernels.memory_layout import get_codec_for_block

        # Sink blocks (0-3) should use Golay
        for i in range(4):
            assert get_codec_for_block(i, sink_blocks=4) == "golay"

        # Context blocks (4+) should use Hamming84
        for i in range(4, 10):
            assert get_codec_for_block(i, sink_blocks=4) == "hamming84"


class TestCacheWrite:
    """Tests for ECC cache write operations."""

    def test_compute_quantization_scales(self):
        """Test quantization scale computation."""
        from vllm_kernels.paged_cache_ecc import compute_quantization_scales

        tensor = torch.tensor([
            [1.0, -2.0, 3.0, -4.0],
            [0.5, -0.5, 0.1, -0.1],
        ], device="cuda")

        scales = compute_quantization_scales(tensor, dim=-1)

        assert scales.shape == (2,)
        # First row: max abs is 4.0, scale = 4.0/7
        assert scales[0].item() == pytest.approx(4.0 / 7, rel=0.01)
        # Second row: max abs is 0.5, scale = 0.5/7
        assert scales[1].item() == pytest.approx(0.5 / 7, rel=0.01)

    def test_write_kv_simple_hamming84(self):
        """Test simple (non-fused) cache write with Hamming84."""
        from vllm_kernels.paged_cache_ecc import write_kv_to_cache_simple

        batch_size = 2
        seq_len = 32
        hidden = 64

        kv = torch.randn(batch_size, seq_len, hidden, device="cuda", dtype=torch.float16)

        encoded, scales = write_kv_to_cache_simple(kv, codec="hamming84")

        assert encoded.shape == kv.shape
        assert encoded.dtype == torch.uint8
        assert scales.shape == (batch_size, seq_len)

    def test_write_kv_roundtrip_hamming84(self):
        """Test encode->decode roundtrip preserves data (with quantization loss)."""
        from vllm_kernels.paged_cache_ecc import (
            write_kv_to_cache_simple, compute_quantization_scales
        )
        from hamming74.triton_kernels import hamming84_decode

        batch_size = 2
        seq_len = 16
        hidden = 32

        kv = torch.randn(batch_size, seq_len, hidden, device="cuda", dtype=torch.float16)
        scales = compute_quantization_scales(kv.float(), dim=-1)

        encoded, _ = write_kv_to_cache_simple(kv, codec="hamming84", scale=scales)

        # Decode
        decoded, stats = hamming84_decode(encoded.flatten())
        decoded = decoded.view(encoded.shape)

        # Dequantize
        dequantized = (decoded.float() - 8) * scales.unsqueeze(-1).cuda()

        # Should be close to original (within quantization error)
        mse = ((kv.float() - dequantized) ** 2).mean()
        assert mse < 1.0, f"MSE too high: {mse}"


class TestBenchmarkHarness:
    """Tests for benchmark harness functions."""

    def test_cuda_timer(self):
        """Test CUDA timer function."""
        from vllm_kernels.benchmark_harness import cuda_timer

        data = torch.randn(1000, device="cuda")

        # Simple operation
        latency = cuda_timer(lambda: data + 1, warmup=5, repeat=50)

        assert latency > 0
        assert latency < 1000  # Should be sub-millisecond

    def test_benchmark_hamming84_encode(self):
        """Test Hamming84 encode benchmark."""
        from vllm_kernels.benchmark_harness import benchmark_hamming84_encode

        result = benchmark_hamming84_encode(n_elements=10_000, warmup=5, repeat=20)

        assert result.name == "hamming84_encode"
        assert result.n_elements == 10_000
        assert result.latency_us > 0
        assert result.throughput_mvals_sec > 0

    def test_benchmark_golay_encode(self):
        """Test Golay encode benchmark."""
        from vllm_kernels.benchmark_harness import benchmark_golay_encode

        result = benchmark_golay_encode(n_triplets=3_333, warmup=5, repeat=20)

        assert result.name == "golay_encode"
        assert result.latency_us > 0
        assert result.throughput_mvals_sec > 0

    def test_benchmark_fault_injection(self):
        """Test fault injection benchmark."""
        from vllm_kernels.benchmark_harness import benchmark_fault_injection

        result = benchmark_fault_injection(n_elements=10_000, ber=0.05, warmup=5, repeat=20)

        assert "fault_injection" in result.name
        assert result.extra["ber"] == 0.05

    def test_benchmark_encode_inject_decode_pipeline(self):
        """Test end-to-end pipeline benchmark."""
        from vllm_kernels.benchmark_harness import benchmark_encode_inject_decode

        result = benchmark_encode_inject_decode(
            codec="hamming84", n_elements=10_000, ber=0.01, warmup=5, repeat=20
        )

        assert "pipeline" in result.name
        assert result.extra["codec"] == "hamming84"


class TestIntegration:
    """Integration tests for full ECC pipeline."""

    def test_full_ecc_pipeline_hamming84(self):
        """Test full encode -> inject -> decode -> verify pipeline."""
        from hamming74.triton_kernels import (
            hamming84_encode, hamming84_decode, inject_bit_errors_triton
        )

        # Create test data
        n_values = 10_000
        original = torch.randint(0, 16, (n_values,), dtype=torch.uint8, device="cuda")

        # Encode
        encoded = hamming84_encode(original)

        # Inject errors (low BER)
        corrupted, stats = inject_bit_errors_triton(
            encoded, ber=0.001, n_bits=8, seed=42, return_stats=True
        )

        # Decode
        decoded, decode_stats = hamming84_decode(corrupted)

        # Most values should be recovered (Hamming can correct 1-bit errors)
        accuracy = (decoded == original).float().mean()
        assert accuracy > 0.99, f"Accuracy too low: {accuracy}"

    def test_full_ecc_pipeline_golay(self):
        """Test full Golay encode -> inject -> decode -> verify pipeline."""
        from hamming74.triton_kernels import (
            golay_encode, golay_decode, inject_bit_errors_triton
        )

        # Create test triplets
        n_triplets = 3_333
        original = torch.randint(0, 16, (n_triplets, 3), dtype=torch.uint8, device="cuda")

        # Encode
        encoded = golay_encode(original)

        # Inject errors (moderate BER - Golay handles up to 3 errors)
        corrupted, stats = inject_bit_errors_triton(
            encoded, ber=0.01, n_bits=24, seed=42, return_stats=True
        )

        # Decode
        decoded, decode_stats = golay_decode(corrupted)

        # Most values should be recovered (Golay can correct up to 3 errors)
        accuracy = (decoded == original).float().mean()
        assert accuracy > 0.98, f"Accuracy too low: {accuracy}"
