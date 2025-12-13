# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements bit-significance-aware error protection for quantized KV caches in LLMs. It provides Error Correcting Code (ECC) schemes integrated with Triton GPU kernels to protect INT4-quantized key-value caches from bit errors during inference.

## Build and Run Commands

### Running Tests
```bash
# Run all tests (requires CUDA)
pytest tests/ -v

# Run specific test file
pytest tests/test_vllm_kernels.py -v

# Run single test
pytest tests/test_vllm_kernels.py::TestMemoryLayout::test_ecc_cache_config_hamming84 -v
```

### Running Benchmarks via Modal (Cloud GPU)
```bash
# Run tests on T4 GPU
modal run modal_runner.py

# Run tests on A100 GPU (for LLaMA)
modal run modal_runner.py --llama-tests

# Quick verification
modal run modal_runner.py --verify

# Latency benchmarks
modal run modal_runner.py --latency

# Attention kernel benchmarks
modal run modal_runner.py --benchmark-kernels

# Triton ECC perplexity sweep (parallel A100 jobs)
modal run modal_runner.py --eval-triton --triton-model "meta-llama/Llama-3.1-8B"

# Full benchmark suite
modal run modal_runner.py --benchmark --models gpt2 --seeds 3

# List/pull benchmark results
modal run modal_runner.py --list-results-flag
modal run modal_runner.py --pull-latest
```

## Architecture

### Core ECC Codecs (`hamming74/`)
- **Hamming74** (`hamming74_sec.py`): Hamming(7,4) single-error correction - encodes 4 data bits into 7-bit codewords
- **Hamming84** (`hamming84_secded.py`): Hamming(8,4) SECDED - adds overall parity for double-error detection
- **Golay2412** (`golay.py`): Golay(24,12) - encodes 12 data bits (3 INT4 values) with 3-bit error correction

### Triton GPU Kernels (`hamming74/triton_kernels/`)
GPU-accelerated implementations of all codecs plus:
- `fault_injection_triton.py`: BER-based bit error injection for testing
- `interpolation_triton.py`: Manifold interpolation for double-error recovery

### vLLM Integration (`vllm_kernels/`)
- `memory_layout.py`: `ECCCacheConfig` for paged KV cache with ECC, block tables, slot mapping
- `attention_ecc.py`: Triton paged attention kernel with inline ECC decode
- `paged_cache_ecc.py`: Quantization and cache write operations
- `benchmark_harness.py`: Latency measurement utilities

### Evaluation Framework (`evaluation/`)
- `sweep.py`: Monte Carlo BER sweep across cache modes
- `metrics.py`: Perplexity, KL divergence, Top-5 accuracy, catastrophic failure rate
- `experiments/monte_carlo.py`: Full experiment runner with result formatting
- `runners/modal.py`: Modal cloud execution helpers
- `runners/triton_eval.py`: Per-trial Triton evaluation

### Cache Modes
The system supports these protection strategies (defined in `evaluation/constants.py`):
- `fp16`: No quantization (baseline)
- `int4`: Quantized, no ECC
- `int4-hamming`: Hamming(7,4) SEC
- `int4-hamming84`: Hamming(8,4) SECDED
- `int4-hamming84-interp`: SECDED + manifold interpolation for uncorrectable errors
- `int12-golay`: Golay(24,12) for 3-bit error correction
- `adaptive`: Golay for sink tokens, Hamming84 for context

### Adaptive UEP (Unequal Error Protection)
Sink tokens (attention sinks at sequence start) get stronger Golay protection via `sink_blocks` parameter in `ECCCacheConfig`. Block codec selection happens in `get_codec_for_block()`.

## Key Implementation Details

- All ECC operations use syndrome decoding with precomputed lookup tables
- Triton kernels use `tl.constexpr` for syndrome LUTs to avoid memory loads
- INT4 quantization uses symmetric range [-8, 7] with per-token scales
- Paged attention kernel decodes ECC inline during attention computation
- Tests require CUDA (`conftest.py` sets `device="cuda"`)
