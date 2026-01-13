import pytest
import torch


class TestAttentionBenchmarkResult:
    def test_dataclass_fields(self):
        from kv_cache.benchmark_harness import AttentionBenchmarkResult

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
    def test_randomized_block_table_shape(self):
        from kv_cache.benchmark_harness import _create_randomized_block_table

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
        from kv_cache.benchmark_harness import _create_randomized_block_table

        batch_size = 2
        num_blocks_per_seq = 5
        total_blocks = 20

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        assert (block_table >= 0).all()
        assert (block_table < total_blocks).all()

        for b in range(batch_size):
            batch_blocks = block_table[b].cpu().tolist()
            assert len(batch_blocks) == len(set(batch_blocks)), "Duplicate blocks found"

    def test_randomized_block_table_is_different_per_batch(self):
        from kv_cache.benchmark_harness import _create_randomized_block_table

        batch_size = 4
        num_blocks_per_seq = 10
        total_blocks = 100

        block_table = _create_randomized_block_table(
            batch_size, num_blocks_per_seq, total_blocks, "cuda"
        )

        different_found = False
        for i in range(batch_size - 1):
            if not torch.equal(block_table[i], block_table[i + 1]):
                different_found = True
                break
        assert (
            different_found
        ), "All batches have same block table - randomization may be broken"


class TestBenchmarkAttentionBaseline:
    def test_benchmark_attention_baseline_returns_result(self):
        from kv_cache.benchmark_harness import benchmark_attention_baseline

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
        from kv_cache.benchmark_harness import benchmark_attention_baseline

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

        assert (
            long_result.latency_us >= short_result.latency_us * 0.5
        ), "Long sequence unexpectedly faster than short"


class TestBenchmarkAttentionECCHamming84:
    def test_benchmark_attention_ecc_hamming84_returns_result(self):
        from kv_cache.benchmark_harness import benchmark_attention_ecc_hamming84

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
        from kv_cache.benchmark_harness import benchmark_attention_ecc_hamming84

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
        from kv_cache.benchmark_harness import benchmark_attention_ecc_hamming84

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


class TestAttentionResultsToJson:
    def test_attention_results_to_json(self):
        import json
        from kv_cache.benchmark_harness import (
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
    def test_prepare_ecc_cache_hamming84_shapes(self):
        from kv_cache.benchmark_harness import (
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
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            num_layers,
            block_size,
            total_blocks,
            block_table,
            "cuda",
        )

        codewords_per_head = block_size * head_dim
        assert k_cache.shape == (
            total_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
        )
        assert v_cache.shape == (
            total_blocks,
            num_layers,
            num_heads,
            codewords_per_head,
        )
        assert scales.shape == (total_blocks, num_layers, num_heads, block_size)
        assert k_cache.dtype == torch.uint8
        assert scales.dtype == torch.float32

    def test_prepare_ecc_cache_hamming84_valid_encoded_data(self):
        from ecc_codecs.triton_kernels import hamming84_decode
        from kv_cache.benchmark_harness import (
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
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            num_layers,
            block_size,
            total_blocks,
            block_table,
            "cuda",
        )

        phys_block = int(block_table[0, 0].item())
        encoded_sample = k_cache[phys_block, 0, 0, :head_dim].flatten()
        decoded, stats = hamming84_decode(encoded_sample)

        assert (decoded >= 0).all()
        assert (decoded <= 15).all()


class TestRunAttentionBenchmarkSuite:
    def test_run_attention_benchmark_suite_minimal(self):
        from kv_cache.benchmark_harness import run_attention_benchmark_suite

        results = run_attention_benchmark_suite(
            batch_sizes=[1],
            seq_lens=[64],
            num_heads=4,
            head_dim=32,
            warmup=2,
            repeat=3,
        )

        assert len(results) == 2

        names = [r.name for r in results]
        assert "pytorch_sdpa_baseline" in names
        assert "paged_attention_ecc_hamming84" in names

        for r in results:
            assert r.latency_us > 0
            assert r.tokens_per_sec > 0
