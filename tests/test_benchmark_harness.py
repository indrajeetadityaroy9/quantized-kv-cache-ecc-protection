"""
Tests for the benchmark harness, specifically attention kernel benchmarks.
"""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestAttentionBenchmarkResult:
    """Tests for AttentionBenchmarkResult dataclass."""

    def test_dataclass_fields(self):
        """Test that all expected fields are present."""
        from vllm_kernels.benchmark_harness import AttentionBenchmarkResult

        result = AttentionBenchmarkResult(
            name="test",
            batch_size=4,
            seq_len=512,
            num_heads=32,
            head_dim=128,
            latency_us=100.0,
            tokens_per_sec=1e6,
            overhead_vs_baseline=1.5,
            extra={"block_size": 16},
        )

        assert result.name == "test"
        assert result.batch_size == 4
        assert result.seq_len == 512
        assert result.num_heads == 32
        assert result.head_dim == 128
        assert result.latency_us == 100.0
        assert result.tokens_per_sec == 1e6
        assert result.overhead_vs_baseline == 1.5
        assert result.extra == {"block_size": 16}


class TestRandomizedBlockTable:
    """Tests for the randomized block table helper."""

    def test_randomized_block_table_shape(self):
        """Test that block table has correct shape."""
        from vllm_kernels.benchmark_harness import _create_randomized_block_table

        batch_size = 4
        num_blocks_per_seq = 8
        total_blocks = 64

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        assert block_table.shape == (batch_size, num_blocks_per_seq)
        assert block_table.dtype == torch.int32
        assert block_table.device.type == "cuda"

    def test_randomized_block_table_values(self):
        """Test that block table contains valid unique block indices."""
        from vllm_kernels.benchmark_harness import _create_randomized_block_table

        batch_size = 2
        num_blocks_per_seq = 5
        total_blocks = 20

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        # All values should be in valid range
        assert (block_table >= 0).all()
        assert (block_table < total_blocks).all()

        # Each batch should have unique block indices (no duplicates within a batch)
        for b in range(batch_size):
            batch_blocks = block_table[b].cpu().tolist()
            assert len(batch_blocks) == len(set(batch_blocks)), "Duplicate blocks found"

    def test_randomized_block_table_is_different_per_batch(self):
        """Test that different batches get different block assignments."""
        from vllm_kernels.benchmark_harness import _create_randomized_block_table

        batch_size = 4
        num_blocks_per_seq = 10
        total_blocks = 100

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        # At least some batches should have different block assignments
        # (very unlikely to be all the same due to random permutation)
        different_found = False
        for i in range(batch_size - 1):
            if not torch.equal(block_table[i], block_table[i + 1]):
                different_found = True
                break
        assert different_found, "All batches have same block table - randomization may be broken"


class TestBenchmarkAttentionBaseline:
    """Tests for the baseline SDPA benchmark."""

    def test_benchmark_attention_baseline_returns_result(self):
        """Test that baseline benchmark returns valid result."""
        from vllm_kernels.benchmark_harness import benchmark_attention_baseline

        result = benchmark_attention_baseline(
            batch_size=1,
            seq_len=64,
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=5,
        )

        assert result.name == "pytorch_sdpa_baseline"
        assert result.batch_size == 1
        assert result.seq_len == 64
        assert result.num_heads == 4
        assert result.head_dim == 32
        assert result.latency_us > 0
        assert result.tokens_per_sec > 0
        assert result.overhead_vs_baseline == 1.0

    def test_benchmark_attention_baseline_latency_scales_with_seq_len(self):
        """Test that latency increases with longer sequences."""
        from vllm_kernels.benchmark_harness import benchmark_attention_baseline

        short_result = benchmark_attention_baseline(
            batch_size=1,
            seq_len=64,
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=10,
        )

        long_result = benchmark_attention_baseline(
            batch_size=1,
            seq_len=256,
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=10,
        )

        # Longer sequences should take more time (not strictly required but expected)
        # Allow for some variance - just check that it's not drastically wrong
        assert long_result.latency_us >= short_result.latency_us * 0.5, \
            "Long sequence unexpectedly faster than short"


class TestBenchmarkAttentionECCHamming84:
    """Tests for the Hamming84 ECC attention benchmark."""

    def test_benchmark_attention_ecc_hamming84_returns_result(self):
        """Test that Hamming84 benchmark returns valid result."""
        from vllm_kernels.benchmark_harness import benchmark_attention_ecc_hamming84

        result = benchmark_attention_ecc_hamming84(
            batch_size=1,
            seq_len=64,
            num_heads=4,
            head_dim=32,
            block_size=16,
            warmup=2,
            repeat=5,
        )

        assert result.name == "paged_attention_ecc_hamming84"
        assert result.batch_size == 1
        assert result.seq_len == 64
        assert result.num_heads == 4
        assert result.head_dim == 32
        assert result.latency_us > 0
        assert result.tokens_per_sec > 0
        assert result.extra["codec"] == "hamming84"

    def test_benchmark_attention_ecc_hamming84_overhead_computed(self):
        """Test that overhead is computed when baseline provided."""
        from vllm_kernels.benchmark_harness import benchmark_attention_ecc_hamming84

        result = benchmark_attention_ecc_hamming84(
            batch_size=1,
            seq_len=64,
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=5,
            baseline_latency_us=10.0,
        )

        assert result.overhead_vs_baseline is not None
        assert result.overhead_vs_baseline > 0

    def test_benchmark_attention_ecc_hamming84_no_overhead_without_baseline(self):
        """Test that overhead is None when baseline not provided."""
        from vllm_kernels.benchmark_harness import benchmark_attention_ecc_hamming84

        result = benchmark_attention_ecc_hamming84(
            batch_size=1,
            seq_len=64,
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=5,
            baseline_latency_us=None,
        )

        assert result.overhead_vs_baseline is None


class TestBenchmarkAttentionECCAdaptive:
    """Tests for the Adaptive UEP attention benchmark."""

    def test_benchmark_attention_ecc_adaptive_returns_result(self):
        """Test that Adaptive UEP benchmark returns valid result."""
        from vllm_kernels.benchmark_harness import benchmark_attention_ecc_adaptive

        result = benchmark_attention_ecc_adaptive(
            batch_size=1,
            seq_len=128,  # Needs to be long enough for sink blocks
            num_heads=4,
            head_dim=32,
            block_size=16,
            sink_blocks=2,
            warmup=2,
            repeat=5,
        )

        assert result.name == "paged_attention_ecc_adaptive"
        assert result.batch_size == 1
        assert result.seq_len == 128
        assert result.latency_us > 0
        assert result.tokens_per_sec > 0
        assert result.extra["sink_blocks"] == 2


class TestAttentionResultsToJson:
    """Tests for JSON serialization of attention results."""

    def test_attention_results_to_json(self):
        """Test JSON serialization."""
        import json
        from vllm_kernels.benchmark_harness import (
            AttentionBenchmarkResult,
            attention_results_to_json,
        )

        results = [
            AttentionBenchmarkResult(
                name="test1",
                batch_size=1,
                seq_len=64,
                num_heads=4,
                head_dim=32,
                latency_us=10.0,
                tokens_per_sec=1e5,
                overhead_vs_baseline=1.0,
            ),
            AttentionBenchmarkResult(
                name="test2",
                batch_size=4,
                seq_len=256,
                num_heads=8,
                head_dim=64,
                latency_us=50.0,
                tokens_per_sec=5e5,
                overhead_vs_baseline=1.5,
                extra={"codec": "hamming84"},
            ),
        ]

        json_str = attention_results_to_json(results)
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["name"] == "test1"
        assert data[0]["latency_us"] == 10.0
        assert data[1]["codec"] == "hamming84"


class TestPrepareECCCacheHamming84:
    """Tests for the Hamming84 cache preparation helper."""

    def test_prepare_ecc_cache_hamming84_shapes(self):
        """Test that prepared cache has correct shapes."""
        from vllm_kernels.benchmark_harness import (
            _create_randomized_block_table,
            _prepare_ecc_cache_hamming84,
        )

        batch_size = 2
        seq_len = 32
        num_heads = 4
        head_dim = 16
        num_layers = 1
        block_size = 16
        num_blocks_per_seq = (seq_len + block_size - 1) // block_size
        total_blocks = batch_size * num_blocks_per_seq * 2

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        k_cache, v_cache, scales = _prepare_ecc_cache_hamming84(
            batch_size, seq_len, num_heads, head_dim, num_layers, block_size,
            total_blocks, block_table, "cuda"
        )

        codewords_per_head = block_size * head_dim
        assert k_cache.shape == (total_blocks, num_layers, num_heads, codewords_per_head)
        assert v_cache.shape == (total_blocks, num_layers, num_heads, codewords_per_head)
        assert scales.shape == (total_blocks, num_layers, num_heads, block_size)
        assert k_cache.dtype == torch.uint8
        assert scales.dtype == torch.float32

    def test_prepare_ecc_cache_hamming84_valid_encoded_data(self):
        """Test that cache contains valid Hamming84 encoded data."""
        from hamming74.triton_kernels import hamming84_decode
        from vllm_kernels.benchmark_harness import (
            _create_randomized_block_table,
            _prepare_ecc_cache_hamming84,
        )

        batch_size = 1
        seq_len = 16
        num_heads = 2
        head_dim = 8
        num_layers = 1
        block_size = 16
        num_blocks_per_seq = 1
        total_blocks = 4

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        k_cache, v_cache, scales = _prepare_ecc_cache_hamming84(
            batch_size, seq_len, num_heads, head_dim, num_layers, block_size,
            total_blocks, block_table, "cuda"
        )

        # Decode some data to verify it's valid Hamming84
        phys_block = int(block_table[0, 0].item())
        encoded_sample = k_cache[phys_block, 0, 0, :head_dim].flatten()
        decoded, stats = hamming84_decode(encoded_sample)

        # Decoded values should be in INT4 range [0, 15]
        assert (decoded >= 0).all()
        assert (decoded <= 15).all()


class TestRunAttentionBenchmarkSuite:
    """Tests for the full benchmark suite runner."""

    def test_run_attention_benchmark_suite_minimal(self):
        """Test running the benchmark suite with minimal config."""
        from vllm_kernels.benchmark_harness import run_attention_benchmark_suite

        results = run_attention_benchmark_suite(
            batch_sizes=[1],
            seq_lens=[64],
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=3,
        )

        # Should have 3 results: baseline, hamming84, adaptive
        assert len(results) == 3

        names = [r.name for r in results]
        assert "pytorch_sdpa_baseline" in names
        assert "paged_attention_ecc_hamming84" in names
        assert "paged_attention_ecc_adaptive" in names

        # All should have positive latency
        for r in results:
            assert r.latency_us > 0
            assert r.tokens_per_sec > 0
