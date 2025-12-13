import json
import time

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from hamming74.hamming84_secded import Hamming84
from hamming74.triton_kernels import (
    golay_decode,
    golay_encode,
    hamming84_decode,
    hamming84_encode,
    inject_bit_errors_triton,
)
from vllm_kernels.attention_ecc import (
    paged_attention_ecc,
    paged_attention_ecc_adaptive,
)


@dataclass
class BenchmarkResult:
    name: str
    n_elements: int
    n_bits: int
    latency_us: float
    throughput_mvals_sec: float
    extra: dict | None = None


@dataclass
class AttentionBenchmarkResult:
    name: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    latency_us: float
    tokens_per_sec: float
    overhead_vs_baseline: float | None = None
    extra: dict | None = None


def cuda_timer(func, warmup=10, repeat=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        func()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms * 1000) / repeat


def benchmark_hamming84_encode(n_elements=1_000_000, warmup=10, repeat=100):
    data = torch.randint(0, 16, (n_elements,), dtype=torch.uint8, device="cuda")

    latency_us = cuda_timer(
        lambda: hamming84_encode(data),
        warmup=warmup,
        repeat=repeat,
    )

    throughput = n_elements / latency_us

    return BenchmarkResult(
        name="hamming84_encode",
        n_elements=n_elements,
        n_bits=8,
        latency_us=latency_us,
        throughput_mvals_sec=throughput,
    )


def benchmark_hamming84_decode(n_elements=1_000_000, warmup=10, repeat=100):
    data = torch.randint(0, 16, (n_elements,), dtype=torch.uint8, device="cuda")
    encoded = hamming84_encode(data)

    latency_us = cuda_timer(
        lambda: hamming84_decode(encoded),
        warmup=warmup,
        repeat=repeat,
    )

    throughput = n_elements / latency_us

    return BenchmarkResult(
        name="hamming84_decode",
        n_elements=n_elements,
        n_bits=8,
        latency_us=latency_us,
        throughput_mvals_sec=throughput,
    )


def benchmark_golay_encode(n_triplets=333_333, warmup=10, repeat=100):
    triplets = torch.randint(0, 16, (n_triplets, 3), dtype=torch.uint8, device="cuda")

    latency_us = cuda_timer(
        lambda: golay_encode(triplets),
        warmup=warmup,
        repeat=repeat,
    )

    n_elements = n_triplets * 3
    throughput = n_elements / latency_us

    return BenchmarkResult(
        name="golay_encode",
        n_elements=n_elements,
        n_bits=24,
        latency_us=latency_us,
        throughput_mvals_sec=throughput,
    )


def benchmark_golay_decode(n_triplets=333_333, warmup=10, repeat=100):
    triplets = torch.randint(0, 16, (n_triplets, 3), dtype=torch.uint8, device="cuda")
    encoded = golay_encode(triplets)

    latency_us = cuda_timer(
        lambda: golay_decode(encoded),
        warmup=warmup,
        repeat=repeat,
    )

    n_elements = n_triplets * 3
    throughput = n_elements / latency_us

    return BenchmarkResult(
        name="golay_decode",
        n_elements=n_elements,
        n_bits=24,
        latency_us=latency_us,
        throughput_mvals_sec=throughput,
    )


def benchmark_fault_injection(
    n_elements=1_000_000, ber=0.01, n_bits=8, warmup=10, repeat=100
):
    if n_bits <= 8:
        data = torch.randint(0, 256, (n_elements,), dtype=torch.uint8, device="cuda")
    else:
        data = torch.randint(
            0, 2**24, (n_elements,), dtype=torch.int32, device="cuda"
        )

    seed = 42

    latency_us = cuda_timer(
        lambda: inject_bit_errors_triton(data, ber, n_bits, seed),
        warmup=warmup,
        repeat=repeat,
    )

    throughput = n_elements / latency_us

    return BenchmarkResult(
        name=f"fault_injection_ber{ber}",
        n_elements=n_elements,
        n_bits=n_bits,
        latency_us=latency_us,
        throughput_mvals_sec=throughput,
        extra={"ber": ber},
    )


def benchmark_encode_inject_decode(
    codec="hamming84", n_elements=1_000_000, ber=0.01, warmup=10, repeat=100
):
    if codec == "hamming84":
        data = torch.randint(0, 16, (n_elements,), dtype=torch.uint8, device="cuda")
        n_bits = 8

        def pipeline():
            encoded = hamming84_encode(data)
            corrupted = inject_bit_errors_triton(encoded, ber, n_bits, seed=42)
            decoded, _ = hamming84_decode(corrupted)
            return decoded

    else:
        n_triplets = (n_elements + 2) // 3
        triplets = torch.randint(
            0, 16, (n_triplets, 3), dtype=torch.uint8, device="cuda"
        )
        n_bits = 24

        def pipeline():
            encoded = golay_encode(triplets)
            corrupted = inject_bit_errors_triton(encoded, ber, n_bits, seed=42)
            decoded, _ = golay_decode(corrupted)
            return decoded

    latency_us = cuda_timer(pipeline, warmup=warmup, repeat=repeat)
    throughput = n_elements / latency_us

    return BenchmarkResult(
        name=f"{codec}_pipeline_ber{ber}",
        n_elements=n_elements,
        n_bits=n_bits,
        latency_us=latency_us,
        throughput_mvals_sec=throughput,
        extra={"codec": codec, "ber": ber},
    )


def _create_randomized_block_table(batch_size, num_blocks_per_seq, total_blocks, device="cuda"):
    block_table = torch.full(
        (batch_size, num_blocks_per_seq),
        -1,
        dtype=torch.int32,
        device=device,
    )

    for b in range(batch_size):
        perm = torch.randperm(total_blocks, device=device)[:num_blocks_per_seq]
        block_table[b, :] = perm.to(torch.int32)

    return block_table


def _prepare_ecc_cache_hamming84(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    num_layers,
    block_size,
    num_blocks,
    block_table,
    device="cuda",
):
    layer_idx = 0
    codewords_per_head = block_size * head_dim

    k_cache = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        codewords_per_head,
        dtype=torch.uint8,
        device=device,
    )
    v_cache = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        codewords_per_head,
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        block_size,
        dtype=torch.float32,
        device=device,
    )

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    for b in range(batch_size):
        for blk_idx in range(num_blocks_per_seq):
            phys_block = int(block_table[b, blk_idx].item())

            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, seq_len)
            tokens_in_block = end_pos - start_pos

            for slot in range(tokens_in_block):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                k_scale = k_fp.abs().max(dim=-1).values / 7.0
                v_scale = v_fp.abs().max(dim=-1).values / 7.0
                scale = torch.maximum(k_scale, v_scale)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )

                k_encoded = hamming84_encode(k_int4.flatten()).view(num_heads, head_dim)
                v_encoded = hamming84_encode(v_int4.flatten()).view(num_heads, head_dim)

                offset_start = slot * head_dim
                offset_end = offset_start + head_dim
                k_cache[phys_block, layer_idx, :, offset_start:offset_end] = k_encoded
                v_cache[phys_block, layer_idx, :, offset_start:offset_end] = v_encoded
                scales[phys_block, layer_idx, :, slot] = scale

    return k_cache, v_cache, scales


def _prepare_ecc_cache_golay(
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    num_layers,
    block_size,
    num_blocks,
    block_table,
    device="cuda",
):
    layer_idx = 0
    codewords_per_slot = (head_dim + 2) // 3
    codewords_per_head = block_size * codewords_per_slot

    k_cache = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        codewords_per_head,
        dtype=torch.int32,
        device=device,
    )
    v_cache = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        codewords_per_head,
        dtype=torch.int32,
        device=device,
    )
    scales = torch.zeros(
        num_blocks,
        num_layers,
        num_heads,
        block_size,
        dtype=torch.float32,
        device=device,
    )

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size

    for b in range(batch_size):
        for blk_idx in range(num_blocks_per_seq):
            phys_block = int(block_table[b, blk_idx].item())

            start_pos = blk_idx * block_size
            end_pos = min(start_pos + block_size, seq_len)
            tokens_in_block = end_pos - start_pos

            for slot in range(tokens_in_block):
                k_fp = torch.randn(num_heads, head_dim, device=device)
                v_fp = torch.randn(num_heads, head_dim, device=device)

                k_scale = k_fp.abs().max(dim=-1).values / 7.0
                v_scale = v_fp.abs().max(dim=-1).values / 7.0
                scale = torch.maximum(k_scale, v_scale)
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)

                k_int4 = (torch.round(k_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )
                v_int4 = (torch.round(v_fp / scale.unsqueeze(-1)).clamp(-8, 7) + 8).to(
                    torch.uint8
                )

                for h in range(num_heads):
                    k_padded = torch.zeros(
                        (codewords_per_slot * 3,), dtype=torch.uint8, device=device
                    )
                    v_padded = torch.zeros(
                        (codewords_per_slot * 3,), dtype=torch.uint8, device=device
                    )
                    k_padded[:head_dim] = k_int4[h]
                    v_padded[:head_dim] = v_int4[h]

                    k_triplets = k_padded.view(-1, 3)
                    v_triplets = v_padded.view(-1, 3)
                    k_encoded = golay_encode(k_triplets)
                    v_encoded = golay_encode(v_triplets)

                    offset_start = slot * codewords_per_slot
                    offset_end = offset_start + codewords_per_slot
                    k_cache[
                        phys_block, layer_idx, h, offset_start:offset_end
                    ] = k_encoded
                    v_cache[
                        phys_block, layer_idx, h, offset_start:offset_end
                    ] = v_encoded

                scales[phys_block, layer_idx, :, slot] = scale

    return k_cache, v_cache, scales


def benchmark_attention_baseline(
    batch_size=4,
    seq_len=512,
    num_heads=32,
    head_dim=128,
    warmup=10,
    repeat=100,
):
    device = "cuda"

    query = torch.randn(
        batch_size, num_heads, 1, head_dim, device=device, dtype=torch.float16
    )
    key = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )
    value = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
    )

    def attention_fn():
        return F.scaled_dot_product_attention(query, key, value)

    latency_us = cuda_timer(attention_fn, warmup=warmup, repeat=repeat)
    tokens_per_sec = (batch_size * seq_len) / (latency_us * 1e-6)

    return AttentionBenchmarkResult(
        name="pytorch_sdpa_baseline",
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        latency_us=latency_us,
        tokens_per_sec=tokens_per_sec,
        overhead_vs_baseline=1.0,
    )


def benchmark_attention_ecc_hamming84(
    batch_size=4,
    seq_len=512,
    num_heads=32,
    head_dim=128,
    block_size=16,
    warmup=10,
    repeat=100,
    baseline_latency_us=None,
):
    device = "cuda"
    torch.manual_seed(42)

    num_layers = 1
    layer_idx = 0
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq * 2

    block_table = _create_randomized_block_table(
        batch_size, num_blocks_per_seq, total_blocks, device
    )

    context_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    k_cache, v_cache, scales = _prepare_ecc_cache_hamming84(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        num_layers,
        block_size,
        total_blocks,
        block_table,
        device,
    )

    query = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=torch.float32
    )

    def attention_fn():
        return paged_attention_ecc(
            query,
            k_cache,
            v_cache,
            block_table,
            context_lens,
            scales,
            layer_idx,
            block_size,
            codec="hamming84",
        )

    latency_us = cuda_timer(attention_fn, warmup=warmup, repeat=repeat)
    tokens_per_sec = (batch_size * seq_len) / (latency_us * 1e-6)

    overhead = None
    if baseline_latency_us is not None and baseline_latency_us > 0:
        overhead = latency_us / baseline_latency_us

    return AttentionBenchmarkResult(
        name="paged_attention_ecc_hamming84",
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        latency_us=latency_us,
        tokens_per_sec=tokens_per_sec,
        overhead_vs_baseline=overhead,
        extra={"block_size": block_size, "codec": "hamming84"},
    )


def benchmark_attention_ecc_adaptive(
    batch_size=4,
    seq_len=512,
    num_heads=32,
    head_dim=128,
    block_size=16,
    sink_blocks=4,
    warmup=10,
    repeat=100,
    baseline_latency_us=None,
):
    device = "cuda"
    torch.manual_seed(42)

    num_layers = 1
    layer_idx = 0
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq * 2

    block_table = _create_randomized_block_table(
        batch_size, num_blocks_per_seq, total_blocks, device
    )

    context_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    k_cache, v_cache, scales = _prepare_ecc_cache_hamming84(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        num_layers,
        block_size,
        total_blocks,
        block_table,
        device,
    )

    sink_k_cache, sink_v_cache, sink_scales = _prepare_ecc_cache_golay(
        batch_size,
        min(seq_len, sink_blocks * block_size),
        num_heads,
        head_dim,
        num_layers,
        block_size,
        total_blocks,
        block_table,
        device,
    )

    query = torch.randn(
        batch_size, num_heads, head_dim, device=device, dtype=torch.float32
    )

    def attention_fn():
        return paged_attention_ecc_adaptive(
            query,
            k_cache,
            v_cache,
            sink_k_cache,
            sink_v_cache,
            block_table,
            context_lens,
            scales,
            sink_scales=sink_scales,
            layer_idx=layer_idx,
            block_size=block_size,
            sink_boundary=sink_blocks,
        )

    latency_us = cuda_timer(attention_fn, warmup=warmup, repeat=repeat)
    tokens_per_sec = (batch_size * seq_len) / (latency_us * 1e-6)

    overhead = None
    if baseline_latency_us is not None and baseline_latency_us > 0:
        overhead = latency_us / baseline_latency_us

    return AttentionBenchmarkResult(
        name="paged_attention_ecc_adaptive",
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        latency_us=latency_us,
        tokens_per_sec=tokens_per_sec,
        overhead_vs_baseline=overhead,
        extra={"block_size": block_size, "sink_blocks": sink_blocks},
    )


def run_attention_benchmark_suite(
    batch_sizes=None,
    seq_lens=None,
    num_heads=32,
    head_dim=128,
    warmup=10,
    repeat=50,
):
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    if seq_lens is None:
        seq_lens = [128, 512, 2048]
    results = []

    print("=" * 70)
    print("Attention Kernel Benchmark Suite")
    print("=" * 70)
    print(f"Config: heads={num_heads}, head_dim={head_dim}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lens}")
    print()

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            print(f"\n[batch={batch_size}, seq_len={seq_len}]")

            baseline = benchmark_attention_baseline(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                warmup=warmup,
                repeat=repeat,
            )
            results.append(baseline)
            print(f"  Baseline (SDPA):    {baseline.latency_us:>8.1f} us")

            hamming84 = benchmark_attention_ecc_hamming84(
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                warmup=warmup,
                repeat=repeat,
                baseline_latency_us=baseline.latency_us,
            )
            results.append(hamming84)
            overhead_str = (
                f"{hamming84.overhead_vs_baseline:.2f}x"
                if hamming84.overhead_vs_baseline
                else "N/A"
            )
            print(
                f"  Hamming84:          {hamming84.latency_us:>8.1f} us ({overhead_str})"
            )

            print(f"  Adaptive UEP:       SKIPPED (uses slow reference impl)")

    print("\n" + "=" * 70)
    print("Summary: Overhead vs Baseline")
    print("=" * 70)

    hamming84_overheads = [
        r.overhead_vs_baseline
        for r in results
        if r.name == "paged_attention_ecc_hamming84" and r.overhead_vs_baseline
    ]

    if hamming84_overheads:
        avg_hamming84 = sum(hamming84_overheads) / len(hamming84_overheads)
        min_hamming84 = min(hamming84_overheads)
        max_hamming84 = max(hamming84_overheads)
        print(
            f"  Hamming84 overhead: avg={avg_hamming84:.2f}x, min={min_hamming84:.2f}x, max={max_hamming84:.2f}x"
        )

    print(f"  Adaptive UEP: SKIPPED (needs fused kernel)")
    print()
    return results


def benchmark_cpu_vs_gpu_hamming84(n_elements=100_000):
    cpu_codec = Hamming84(device="cuda")
    data_cpu = torch.randint(0, 16, (n_elements,), dtype=torch.uint8)

    _ = cpu_codec.encode(data_cpu[:1000])

    start = time.perf_counter()
    _ = cpu_codec.encode(data_cpu)
    cpu_encode_time = (time.perf_counter() - start) * 1e6

    encoded_cpu = cpu_codec.encode(data_cpu)
    start = time.perf_counter()
    _ = cpu_codec.decode(encoded_cpu)
    cpu_decode_time = (time.perf_counter() - start) * 1e6

    data_gpu = data_cpu.cuda()

    gpu_encode_time = cuda_timer(
        lambda: hamming84_encode(data_gpu),
        warmup=10,
        repeat=100,
    )

    encoded_gpu = hamming84_encode(data_gpu)
    gpu_decode_time = cuda_timer(
        lambda: hamming84_decode(encoded_gpu),
        warmup=10,
        repeat=100,
    )

    return {
        "cpu_encode": BenchmarkResult(
            name="cpu_hamming84_encode",
            n_elements=n_elements,
            n_bits=8,
            latency_us=cpu_encode_time,
            throughput_mvals_sec=n_elements / cpu_encode_time,
        ),
        "cpu_decode": BenchmarkResult(
            name="cpu_hamming84_decode",
            n_elements=n_elements,
            n_bits=8,
            latency_us=cpu_decode_time,
            throughput_mvals_sec=n_elements / cpu_decode_time,
        ),
        "gpu_encode": BenchmarkResult(
            name="gpu_hamming84_encode",
            n_elements=n_elements,
            n_bits=8,
            latency_us=gpu_encode_time,
            throughput_mvals_sec=n_elements / gpu_encode_time,
        ),
        "gpu_decode": BenchmarkResult(
            name="gpu_hamming84_decode",
            n_elements=n_elements,
            n_bits=8,
            latency_us=gpu_decode_time,
            throughput_mvals_sec=n_elements / gpu_decode_time,
        ),
    }


def run_full_benchmark(sizes=None, ber_levels=None):
    if sizes is None:
        sizes = [10_000, 100_000, 1_000_000]
    if ber_levels is None:
        ber_levels = [0.0, 0.01, 0.05]
    results = []

    print("=" * 70)
    print("Triton ECC Kernel Benchmark Suite")
    print("=" * 70)

    print("\n[Hamming84 Benchmarks]")
    for size in sizes:
        result = benchmark_hamming84_encode(n_elements=size)
        results.append(result)
        print(
            f"  Encode {size:>10,}: {result.latency_us:>8.1f} us, "
            f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
        )

        result = benchmark_hamming84_decode(n_elements=size)
        results.append(result)
        print(
            f"  Decode {size:>10,}: {result.latency_us:>8.1f} us, "
            f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
        )

    print("\n[Golay Benchmarks]")
    for size in sizes:
        n_triplets = size // 3
        result = benchmark_golay_encode(n_triplets=n_triplets)
        results.append(result)
        print(
            f"  Encode {result.n_elements:>10,}: {result.latency_us:>8.1f} us, "
            f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
        )

        result = benchmark_golay_decode(n_triplets=n_triplets)
        results.append(result)
        print(
            f"  Decode {result.n_elements:>10,}: {result.latency_us:>8.1f} us, "
            f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
        )

    print("\n[Fault Injection Benchmarks]")
    for ber in ber_levels:
        result = benchmark_fault_injection(n_elements=1_000_000, ber=ber)
        results.append(result)
        print(
            f"  BER={ber:.2f}: {result.latency_us:>8.1f} us, "
            f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
        )

    print("\n[End-to-End Pipeline Benchmarks]")
    for codec in ["hamming84", "golay"]:
        for ber in [0.0, 0.01]:
            result = benchmark_encode_inject_decode(
                codec=codec, n_elements=1_000_000, ber=ber
            )
            results.append(result)
            print(
                f"  {codec:>10} BER={ber:.2f}: {result.latency_us:>8.1f} us, "
                f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
            )

    print("\n[CPU vs GPU Comparison (100K elements)]")
    comparison = benchmark_cpu_vs_gpu_hamming84(n_elements=100_000)
    for key, result in comparison.items():
        results.append(result)
        print(
            f"  {result.name:>25}: {result.latency_us:>10.1f} us, "
            f"{result.throughput_mvals_sec:>8.1f} Mvals/s"
        )

    cpu_encode = comparison["cpu_encode"].latency_us
    gpu_encode = comparison["gpu_encode"].latency_us
    cpu_decode = comparison["cpu_decode"].latency_us
    gpu_decode = comparison["gpu_decode"].latency_us

    print(f"\n  Encode speedup: {cpu_encode / gpu_encode:.1f}x")
    print(f"  Decode speedup: {cpu_decode / gpu_decode:.1f}x")

    print("\n" + "=" * 70)
    print("Benchmark complete!")

    return results


def results_to_json(results):
    data = []
    for r in results:
        entry = {
            "name": r.name,
            "n_elements": r.n_elements,
            "n_bits": r.n_bits,
            "latency_us": r.latency_us,
            "throughput_mvals_sec": r.throughput_mvals_sec,
        }
        if r.extra:
            entry.update(r.extra)
        data.append(entry)
    return json.dumps(data, indent=2)


def attention_results_to_json(results):
    data = []
    for r in results:
        entry = {
            "name": r.name,
            "batch_size": r.batch_size,
            "seq_len": r.seq_len,
            "num_heads": r.num_heads,
            "head_dim": r.head_dim,
            "latency_us": r.latency_us,
            "tokens_per_sec": r.tokens_per_sec,
            "overhead_vs_baseline": r.overhead_vs_baseline,
        }
        if r.extra:
            entry.update(r.extra)
        data.append(entry)
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    results = run_full_benchmark()
    print("\nJSON Results:")
    print(results_to_json(results))
