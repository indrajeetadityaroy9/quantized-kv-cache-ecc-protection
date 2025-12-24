#!/usr/bin/env python3
"""Comprehensive ICML-quality evaluation of ECC-protected KV cache modes.

This script evaluates ECC-protected INT4 quantized KV caches under various
bit error rates (BER) to demonstrate error correction effectiveness.

Key Metrics:
1. Perplexity (pre/post fault injection comparison)
2. ECC Correction Statistics (errors corrected, uncorrectable)
3. Memory Efficiency (bits per value, compression ratio)
4. Throughput (encode/decode operations per second)
5. Catastrophic Failure Rate (samples with PPL > threshold)

Cache Modes Evaluated:
- FP16 (Oracle baseline)
- INT4 Unprotected (no ECC)
- INT4 + Hamming(7,4) SEC
- INT4 + Hamming(8,4) SECDED
- INT4 + Golay+Hamming Hybrid
- INT4 + Reed-Solomon(12,8)

Usage:
    python run_comprehensive_eval.py  # Full automatic evaluation
"""

import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# Disable V1 multiprocessing to allow direct KV cache access for fault injection
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

import torch
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

# Mapping from evaluation cache modes to vLLM kv_cache_dtype
CACHE_MODE_TO_VLLM_DTYPE = {
    "fp16": "auto",
    "int4": "int4_unprotected",           # Unprotected INT4 (no ECC)
    "int4-hamming": "int4_h74",            # INT4 + Hamming(7,4) SEC
    "int4-hamming84": "int4_ecc",          # INT4 + Hamming(8,4) SECDED
    "int4-golay-hybrid": "int4_golay_hybrid",  # Golay(24,12) + Hamming hybrid
    "int4-reed-solomon": "int4_rs",        # Reed-Solomon(12,8) GF(2^4)
}

CACHE_MODE_LABELS = {
    "fp16": "FP16 (Oracle)",
    "int4": "INT4 (Unprotected)",
    "int4-hamming": "Hamming(7,4) SEC",
    "int4-hamming84": "Hamming(8,4) SECDED",
    "int4-golay-hybrid": "Golay+Hamming Hybrid",
    "int4-reed-solomon": "Reed-Solomon(12,8)",
}

# ECC capabilities and memory characteristics
ECC_CAPABILITIES = {
    "fp16": {
        "bits_per_value": 16,
        "data_bits": 16,
        "ecc_bits": 0,
        "correction_capability": 0,
        "detection_capability": 0,
        "description": "FP16 Oracle baseline (no quantization)",
    },
    "int4": {
        "bits_per_value": 4,
        "data_bits": 4,
        "ecc_bits": 0,
        "correction_capability": 0,
        "detection_capability": 0,
        "description": "INT4 quantized (no error protection)",
    },
    "int4-hamming": {
        "bits_per_value": 7,
        "data_bits": 4,
        "ecc_bits": 3,
        "correction_capability": 1,
        "detection_capability": 1,
        "description": "INT4 + Hamming(7,4) SEC",
    },
    "int4-hamming84": {
        "bits_per_value": 8,
        "data_bits": 4,
        "ecc_bits": 4,
        "correction_capability": 1,
        "detection_capability": 2,
        "description": "INT4 + Hamming(8,4) SECDED",
    },
    "int4-golay-hybrid": {
        "bits_per_value": 8,  # Average: 24 bits for 3 values = 8 bits/value
        "data_bits": 4,
        "ecc_bits": 4,
        "correction_capability": 3,
        "detection_capability": 3,
        "description": "Golay(24,12) for triplets + Hamming(8,4) for remainder",
    },
    "int4-reed-solomon": {
        "bits_per_value": 6,  # 48 bits for 8 values = 6 bits/value
        "data_bits": 4,
        "ecc_bits": 2,
        "correction_capability": 8,  # 2 symbols = 8 bits
        "detection_capability": 8,
        "description": "Reed-Solomon(12,8) over GF(2^4)",
    },
}

# BER levels for evaluation
BER_LEVELS = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# All cache modes to evaluate
ALL_CACHE_MODES = [
    "fp16",
    "int4",
    "int4-hamming",
    "int4-hamming84",
    "int4-golay-hybrid",
    "int4-reed-solomon",
]


@dataclass
class EvalConfig:
    """Evaluation configuration for ICML paper."""
    model: str = "gpt2"
    dataset: str = "wikitext2"
    max_samples: int = 50
    max_length: int = 256
    stride: int = 128
    seeds: List[int] = field(default_factory=lambda: [42])
    ber_levels: List[float] = field(default_factory=lambda: [0, 1e-5, 1e-4, 1e-3])
    cache_modes: List[str] = field(default_factory=lambda: ALL_CACHE_MODES)
    gpu_memory_utilization: float = 0.5
    output_file: str = "icml_eval_results.json"
    throughput_iterations: int = 100
    catastrophic_threshold: float = 1000.0


# ============================================================================
# Data Loading
# ============================================================================

def load_evaluation_texts(dataset: str, max_samples: int) -> List[str]:
    """Load evaluation texts from specified dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets not available, using dummy data")
        return [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "In the beginning, there was nothing but the void. " * 20,
            "Machine learning is transforming how we build software. " * 20,
        ][:max_samples]

    print(f"Loading {dataset} dataset...")

    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for item in ds:
            if len(item["text"].split()) > 50:
                texts.append(item["text"])
            if len(texts) >= max_samples:
                break
    elif dataset == "ptb":
        try:
            ds = load_dataset("ptb_text_only", split="test", trust_remote_code=True)
            current = ""
            texts = []
            for item in ds:
                current += " " + item["sentence"]
                if len(current.split()) > 100:
                    texts.append(current.strip())
                    current = ""
                if len(texts) >= max_samples:
                    break
        except Exception:
            print("Warning: PTB unavailable, falling back to wikitext2")
            return load_evaluation_texts("wikitext2", max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return texts[:max_samples]


# ============================================================================
# Perplexity Computation
# ============================================================================

def compute_perplexity_vllm(
    llm,
    texts: List[str],
    max_length: int = 256,
) -> Tuple[float, List[float]]:
    """Compute perplexity using vLLM's prompt_logprobs.

    Returns:
        Tuple of (mean_perplexity, per_sample_perplexities)
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        temperature=0.0,
    )

    tokenizer = llm.get_tokenizer()
    per_sample_ppls = []

    for text in texts:
        if not text.strip():
            per_sample_ppls.append(float("inf"))
            continue

        tokens = tokenizer.encode(text)
        seq_len = len(tokens)

        if seq_len < 2:
            per_sample_ppls.append(float("inf"))
            continue

        window_tokens = tokens[:min(seq_len, max_length)]

        try:
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": window_tokens}],
                sampling_params=sampling_params,
            )

            if outputs and outputs[0].prompt_logprobs:
                prompt_logprobs = outputs[0].prompt_logprobs
                total_nll = 0.0
                count = 0

                for i in range(1, len(prompt_logprobs)):
                    logprob_dict = prompt_logprobs[i]
                    if logprob_dict is None:
                        continue
                    token_id = window_tokens[i]
                    if token_id in logprob_dict:
                        total_nll -= logprob_dict[token_id].logprob
                        count += 1

                if count > 0:
                    per_sample_ppls.append(math.exp(total_nll / count))
                else:
                    per_sample_ppls.append(float("inf"))
            else:
                per_sample_ppls.append(float("inf"))

        except Exception as e:
            print(f"  Warning: Error computing perplexity: {e}")
            per_sample_ppls.append(float("inf"))

    valid_ppls = [p for p in per_sample_ppls if not math.isinf(p)]
    mean_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("inf")

    return mean_ppl, per_sample_ppls


def inject_faults_into_kv_cache(llm, ber: float, seed: int = 42) -> bool:
    """Inject bit errors into KV cache tensors via InprocClient path.

    With VLLM_ENABLE_V1_MULTIPROCESSING=0, we can access KV cache tensors
    directly through the InprocClient's nested engine_core.

    Access path for InprocClient:
    - llm.llm_engine.engine_core (InprocClient)
    - llm.llm_engine.engine_core.engine_core (EngineCore)
    - llm.llm_engine.engine_core.engine_core.model_executor (Executor)
    - llm.llm_engine.engine_core.engine_core.model_executor.worker (GPUWorker)
    - llm.llm_engine.engine_core.engine_core.model_executor.worker.model_runner (GPUModelRunner)
    - llm.llm_engine.engine_core.engine_core.model_executor.worker.model_runner.kv_caches (list)

    Args:
        llm: vLLM LLM instance (must be using InprocClient mode)
        ber: Bit error rate for fault injection
        seed: Random seed for reproducibility

    Returns:
        True if fault injection was successful, False otherwise
    """
    if ber <= 0:
        return True  # No faults to inject

    try:
        # Access KV cache through InprocClient path
        # InprocClient has .engine_core which is the actual EngineCore
        inproc_client = llm.llm_engine.engine_core
        if hasattr(inproc_client, 'engine_core'):
            # InprocClient wraps EngineCore
            engine_core = inproc_client.engine_core
            model_executor = engine_core.model_executor
        elif hasattr(llm.llm_engine, 'model_executor'):
            # Direct access via LLMEngine shortcut
            model_executor = llm.llm_engine.model_executor
        else:
            print(f"  Could not access model_executor from engine type: {type(llm.llm_engine)}")
            return False

        # Get the worker (for single GPU, this is straightforward)
        # UniProcExecutor uses driver_worker -> worker -> model_runner
        if hasattr(model_executor, 'driver_worker'):
            driver_worker = model_executor.driver_worker
            if hasattr(driver_worker, 'worker'):
                worker = driver_worker.worker
            else:
                worker = driver_worker
        elif hasattr(model_executor, 'worker'):
            worker = model_executor.worker
        elif hasattr(model_executor, 'workers') and model_executor.workers:
            worker = model_executor.workers[0]
        else:
            print(f"  Could not access worker from executor type: {type(model_executor)}")
            return False

        model_runner = worker.model_runner
        kv_caches = model_runner.kv_caches

        if not kv_caches:
            print("  No KV cache tensors found")
            return False

        # Inject faults into each layer's KV cache
        import torch
        total_flipped = 0

        for layer_idx, kv_cache in enumerate(kv_caches):
            if kv_cache is None:
                continue

            # kv_cache is typically [num_blocks, block_size, num_heads, head_dim] or similar
            # For ECC modes, it's stored as uint8
            if kv_cache.dtype == torch.uint8:
                # Use the CUDA fault injection kernel
                try:
                    torch.ops._C_cache_ops.inject_cache_errors(
                        kv_cache,  # key_cache (or combined K,V)
                        kv_cache,  # value_cache (same tensor for now)
                        float(ber),
                        seed + layer_idx
                    )
                    total_flipped += int(kv_cache.numel() * 8 * ber)  # Estimated
                except Exception as e:
                    # Fallback: manual bit flip
                    flat = kv_cache.view(-1)
                    n_bits = flat.numel() * 8
                    n_flips = max(1, int(n_bits * ber))

                    torch.manual_seed(seed + layer_idx)
                    flip_indices = torch.randint(0, flat.numel(), (n_flips,), device=kv_cache.device)
                    flip_bits = torch.randint(0, 8, (n_flips,), device=kv_cache.device)

                    for idx, bit in zip(flip_indices.tolist(), flip_bits.tolist()):
                        flat[idx] = flat[idx] ^ (1 << bit)

                    total_flipped += n_flips

        return True

    except AttributeError as e:
        print(f"  Cannot access KV cache (not InprocClient mode?): {e}")
        return False
    except Exception as e:
        print(f"  Error injecting faults: {e}")
        return False


def compute_perplexity_with_fault_injection(
    llm,
    texts: List[str],
    ber: float,
    seed: int = 42,
    max_length: int = 256,
    context_ratio: float = 0.5,
) -> Tuple[float, List[float], Dict[str, Any]]:
    """Compute perplexity with fault injection into KV cache.

    This function measures perplexity under bit errors by:
    1. Processing context portion of text to populate KV cache (via prefix caching)
    2. Injecting bit errors at specified BER into the cached tensors
    3. Computing perplexity on full sequence (reusing corrupted prefix cache)

    Requires:
    - VLLM_ENABLE_V1_MULTIPROCESSING=0 (enables InprocClient for direct tensor access)
    - enable_prefix_caching=True (allows cache reuse between requests)

    Args:
        llm: vLLM LLM instance
        texts: List of text samples
        ber: Bit error rate for fault injection
        seed: Random seed for reproducibility
        max_length: Maximum sequence length
        context_ratio: Fraction of tokens used as context (rest is continuation)

    Returns:
        Tuple of (mean_perplexity, per_sample_perplexities, stats_dict)
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    per_sample_ppls = []
    stats = {
        "total_samples": 0,
        "valid_samples": 0,
        "skipped_too_short": 0,
        "errors": 0,
        "injection_success": 0,
        "injection_failed": 0,
    }

    # Check if this cache mode supports fault injection
    cache_dtype = llm.llm_engine.cache_config.cache_dtype
    ecc_types = ("int4_ecc", "int4_h74", "int4_golay", "int4_golay_hybrid",
                 "int4_unprotected", "int4_rs")

    can_inject = cache_dtype in ecc_types

    for text in texts:
        stats["total_samples"] += 1

        if not text.strip():
            per_sample_ppls.append(float("inf"))
            continue

        tokens = tokenizer.encode(text)
        seq_len = len(tokens)

        # Need enough tokens for context + continuation
        min_tokens = 20
        if seq_len < min_tokens:
            per_sample_ppls.append(float("inf"))
            stats["skipped_too_short"] += 1
            continue

        # Limit to max_length
        tokens = tokens[:min(seq_len, max_length)]
        seq_len = len(tokens)

        # Split into context and continuation
        context_len = max(10, int(seq_len * context_ratio))
        context_tokens = tokens[:context_len]
        continuation_tokens = tokens[context_len:]

        if len(continuation_tokens) < 5:
            per_sample_ppls.append(float("inf"))
            stats["skipped_too_short"] += 1
            continue

        try:
            # Step 1: Process context to populate KV cache (prefix caching will cache this)
            context_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
            )
            llm.generate(
                prompts=[{"prompt_token_ids": context_tokens}],
                sampling_params=context_params,
            )

            # Step 2: Inject faults into KV cache tensors via InprocClient path
            if can_inject and ber > 0:
                injection_success = inject_faults_into_kv_cache(llm, ber, seed)
                if injection_success:
                    stats["injection_success"] += 1
                else:
                    stats["injection_failed"] += 1

            # Step 3: Compute perplexity on full sequence
            # With prefix caching, the context portion will reuse the (now corrupted) cache
            full_tokens = tokens
            continuation_params = SamplingParams(
                prompt_logprobs=1,
                max_tokens=1,
                temperature=0.0,
            )

            outputs = llm.generate(
                prompts=[{"prompt_token_ids": full_tokens}],
                sampling_params=continuation_params,
            )

            if outputs and outputs[0].prompt_logprobs:
                prompt_logprobs = outputs[0].prompt_logprobs
                total_nll = 0.0
                count = 0

                # Only score continuation tokens (after context)
                for i in range(context_len, len(prompt_logprobs)):
                    logprob_dict = prompt_logprobs[i]
                    if logprob_dict is None:
                        continue
                    token_id = full_tokens[i]
                    if token_id in logprob_dict:
                        total_nll -= logprob_dict[token_id].logprob
                        count += 1

                if count > 0:
                    ppl = math.exp(total_nll / count)
                    per_sample_ppls.append(ppl)
                    stats["valid_samples"] += 1
                else:
                    per_sample_ppls.append(float("inf"))
            else:
                per_sample_ppls.append(float("inf"))

        except Exception as e:
            print(f"  Warning: Error in fault injection PPL: {e}")
            per_sample_ppls.append(float("inf"))
            stats["errors"] += 1

    valid_ppls = [p for p in per_sample_ppls if not math.isinf(p)]
    mean_ppl = sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("inf")

    return mean_ppl, per_sample_ppls, stats


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float, float]:
    """Compute mean, std, and confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    arr = np.array(values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    if len(arr) < 2:
        return mean, std, mean, mean

    from scipy import stats
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=len(arr) - 1)
    margin = t_critical * std / np.sqrt(len(arr))

    return mean, std, mean - margin, mean + margin


def compute_catastrophic_rate(
    perplexities: List[float],
    threshold: float = 1000.0
) -> float:
    """Compute fraction of samples with catastrophic perplexity."""
    if not perplexities:
        return 0.0

    catastrophic = sum(1 for p in perplexities if p > threshold or math.isinf(p))
    return catastrophic / len(perplexities)


# ============================================================================
# Memory Efficiency Metrics
# ============================================================================

def compute_memory_metrics(cache_modes: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute memory efficiency metrics for each cache mode.

    Returns dict with:
    - bits_per_value: Total bits used per KV value
    - compression_ratio: Ratio vs FP16 (higher = more compressed)
    - memory_overhead_pct: ECC overhead as percentage
    """
    results = {}
    fp16_bits = 16

    for mode in cache_modes:
        caps = ECC_CAPABILITIES.get(mode, {})
        bits_per_value = caps.get("bits_per_value", 16)
        data_bits = caps.get("data_bits", bits_per_value)
        ecc_bits = caps.get("ecc_bits", 0)

        compression_ratio = fp16_bits / bits_per_value
        memory_overhead_pct = (ecc_bits / data_bits * 100) if data_bits > 0 else 0

        results[mode] = {
            "bits_per_value": bits_per_value,
            "data_bits": data_bits,
            "ecc_bits": ecc_bits,
            "compression_ratio": compression_ratio,
            "memory_overhead_pct": memory_overhead_pct,
            "bytes_per_value": bits_per_value / 8,
        }

    return results


def compute_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_size: int,
    seq_len: int,
    batch_size: int,
    cache_mode: str,
) -> Dict[str, float]:
    """Compute KV cache size in bytes for a given configuration."""
    caps = ECC_CAPABILITIES.get(cache_mode, {})
    bits_per_value = caps.get("bits_per_value", 16)

    # K and V each: [batch, num_heads, seq_len, head_size]
    num_values = 2 * batch_size * num_layers * num_heads * seq_len * head_size

    total_bits = num_values * bits_per_value
    total_bytes = total_bits / 8

    return {
        "num_values": num_values,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }


# ============================================================================
# Throughput Measurement
# ============================================================================

def measure_kernel_throughput(
    cache_mode: str,
    num_blocks: int = 256,
    block_size: int = 16,
    num_heads: int = 12,
    head_size: int = 64,
    iterations: int = 100,
) -> Dict[str, float]:
    """Measure encode/decode throughput at kernel level.

    Returns dict with:
    - encode_throughput_mvalues_sec: Million values encoded per second
    - decode_throughput_mvalues_sec: Million values decoded per second
    - encode_latency_us: Microseconds per encode operation
    - decode_latency_us: Microseconds per decode operation
    """
    try:
        from vllm._custom_ops import (
            reshape_and_cache_flash,
            cp_gather_and_ecc_decode_kv_cache,
        )
    except ImportError as e:
        return {"error": str(e)}

    kv_cache_dtype = CACHE_MODE_TO_VLLM_DTYPE.get(cache_mode, "auto")

    # Skip non-quantized modes
    if kv_cache_dtype == "auto":
        return {"skipped": "FP16 mode - no custom kernels"}

    num_tokens = num_blocks * block_size // 2

    # Create test tensors
    k = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')
    v = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')

    k_scale = torch.tensor(torch.abs(k).max().item() / 7.0, dtype=torch.float32, device='cuda')
    v_scale = torch.tensor(torch.abs(v).max().item() / 7.0, dtype=torch.float32, device='cuda')

    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device='cuda')

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device='cuda')

    # Warmup
    for _ in range(10):
        reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
    torch.cuda.synchronize()

    # Measure encode
    start = time.perf_counter()
    for _ in range(iterations):
        reshape_and_cache_flash(k, v, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
    torch.cuda.synchronize()
    encode_time = (time.perf_counter() - start) / iterations

    # Setup for decode
    decoded_k = torch.empty(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')
    decoded_v = torch.empty_like(decoded_k)

    blocks_needed = (num_tokens + block_size - 1) // block_size
    block_table = torch.arange(blocks_needed, dtype=torch.int32, device='cuda').unsqueeze(0)
    cu_seq_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda')
    seq_starts = torch.tensor([0], dtype=torch.int32, device='cuda')
    hamming_stats = torch.zeros(4, dtype=torch.int64, device='cuda')

    # Warmup decode
    for _ in range(10):
        cp_gather_and_ecc_decode_kv_cache(
            key_cache, value_cache, decoded_k, decoded_v,
            block_table, cu_seq_lens, seq_starts,
            k_scale, v_scale, kv_cache_dtype,
            None, None, hamming_stats, None
        )
    torch.cuda.synchronize()

    # Measure decode
    start = time.perf_counter()
    for _ in range(iterations):
        cp_gather_and_ecc_decode_kv_cache(
            key_cache, value_cache, decoded_k, decoded_v,
            block_table, cu_seq_lens, seq_starts,
            k_scale, v_scale, kv_cache_dtype,
            None, None, hamming_stats, None
        )
    torch.cuda.synchronize()
    decode_time = (time.perf_counter() - start) / iterations

    num_values = num_tokens * num_heads * head_size * 2  # K and V

    return {
        "encode_throughput_mvalues_sec": (num_values / 1e6) / encode_time,
        "decode_throughput_mvalues_sec": (num_values / 1e6) / decode_time,
        "encode_latency_us": encode_time * 1e6,
        "decode_latency_us": decode_time * 1e6,
        "num_values": num_values,
    }


# ============================================================================
# Fault Injection Testing (Kernel Level)
# ============================================================================

def test_fp16_fault_injection(
    ber_levels: List[float],
    seed: int = 42,
    num_blocks: int = 100,
    block_size: int = 16,
    num_heads: int = 12,
    head_size: int = 64,
) -> Dict[str, Any]:
    """Test FP16 cache degradation under fault injection (no ECC protection).

    This demonstrates that FP16 values are highly sensitive to bit flips,
    providing the baseline to compare against ECC-protected modes.
    """
    results = {}

    try:
        from vllm._custom_ops import inject_cache_errors
    except ImportError as e:
        return {"error": str(e)}

    # Create test data - simulate FP16 KV cache
    num_tokens = num_blocks * block_size // 2
    original_k = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')
    original_v = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')

    # For FP16, there's no quantization error - baseline RMSE is 0
    results["baseline"] = {
        "ber": 0,
        "k_rmse": 0.0,
        "v_rmse": 0.0,
        "bits_flipped": 0,
        "description": "FP16 has no quantization error but no error protection",
    }

    # Test each BER level by directly flipping bits in FP16 representation
    for ber in ber_levels:
        if ber == 0:
            continue

        ber_key = f"ber_{ber:.0e}"

        # Clone original values
        corrupted_k = original_k.clone()
        corrupted_v = original_v.clone()

        # Reinterpret as uint8 for bit flipping (view as bytes)
        k_bytes = corrupted_k.view(torch.uint8)
        v_bytes = corrupted_v.view(torch.uint8)

        k_before = k_bytes.clone()
        v_before = v_bytes.clone()

        # Use the same inject_cache_errors kernel (works on uint8 tensors)
        # Need to reshape to match expected format [num_blocks, block_size, num_heads, head_size]
        k_reshaped = k_bytes.view(num_blocks // 2, block_size, num_heads, head_size * 2)
        v_reshaped = v_bytes.view(num_blocks // 2, block_size, num_heads, head_size * 2)

        inject_cache_errors(k_reshaped, v_reshaped, ber, seed)
        torch.cuda.synchronize()

        # Count actual bits flipped
        k_xor = (k_bytes ^ k_before).flatten().cpu()
        v_xor = (v_bytes ^ v_before).flatten().cpu()
        bits_flipped = 0
        for byte in k_xor.tolist():
            bits_flipped += bin(byte).count('1')
        for byte in v_xor.tolist():
            bits_flipped += bin(byte).count('1')

        # Compute RMSE after corruption - handle NaN/Inf values
        # NaN/Inf in FP16 happens when exponent bits are corrupted
        k_nan_mask = torch.isnan(corrupted_k) | torch.isinf(corrupted_k)
        v_nan_mask = torch.isnan(corrupted_v) | torch.isinf(corrupted_v)
        k_nan_count = k_nan_mask.sum().item()
        v_nan_count = v_nan_mask.sum().item()
        total_values = corrupted_k.numel() + corrupted_v.numel()
        nan_inf_rate = (k_nan_count + v_nan_count) / total_values

        # Compute RMSE on valid (finite) values only - use FP32 to avoid overflow
        k_valid = ~k_nan_mask
        v_valid = ~v_nan_mask

        if k_valid.sum() > 0:
            k_diff = (corrupted_k[k_valid].float() - original_k[k_valid].float()) ** 2
            k_rmse = torch.sqrt(k_diff.mean()).item()
        else:
            k_rmse = float('inf')

        if v_valid.sum() > 0:
            v_diff = (corrupted_v[v_valid].float() - original_v[v_valid].float()) ** 2
            v_rmse = torch.sqrt(v_diff.mean()).item()
        else:
            v_rmse = float('inf')

        total_bits = k_bytes.numel() * 8 * 2

        results[ber_key] = {
            "ber": ber,
            "bits_flipped": bits_flipped,
            "total_bits": total_bits,
            "measured_ber": bits_flipped / total_bits if total_bits > 0 else 0,
            "k_rmse": k_rmse,
            "v_rmse": v_rmse,
            "nan_inf_count": k_nan_count + v_nan_count,
            "nan_inf_rate": nan_inf_rate,
            "rmse_degradation_pct": float("inf"),  # Infinite degradation from 0 baseline
            "correction_rate": 0.0,  # No error correction
            "description": f"FP16 vulnerable: {k_nan_count + v_nan_count} values became NaN/Inf",
        }

    return results


def test_fault_injection_kernel(
    cache_mode: str,
    ber_levels: List[float],
    seed: int = 42,
    num_blocks: int = 100,
    block_size: int = 16,
    num_heads: int = 12,
    head_size: int = 64,
) -> Dict[str, Any]:
    """Test fault injection and ECC correction at CUDA kernel level.

    This directly tests ECC encode/decode kernels with fault injection
    to measure correction effectiveness and reconstruction quality.
    """
    results = {}

    try:
        from vllm._custom_ops import (
            reshape_and_cache_flash,
            cp_gather_and_ecc_decode_kv_cache,
            inject_cache_errors
        )
    except ImportError as e:
        return {"error": str(e)}

    kv_cache_dtype = CACHE_MODE_TO_VLLM_DTYPE.get(cache_mode, "auto")

    # Skip non-ECC modes for kernel testing
    if kv_cache_dtype == "auto":
        return {"skipped": "FP16 mode - no kernel testing"}

    # Create test data
    num_tokens = num_blocks * block_size // 2
    original_k = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')
    original_v = torch.randn(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')

    k_scale = torch.tensor(torch.abs(original_k).max().item() / 7.0, dtype=torch.float32, device='cuda')
    v_scale = torch.tensor(torch.abs(original_v).max().item() / 7.0, dtype=torch.float32, device='cuda')

    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.uint8, device='cuda')

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device='cuda')

    # Encode to cache
    reshape_and_cache_flash(
        original_k, original_v,
        key_cache, value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale, v_scale
    )
    torch.cuda.synchronize()

    # Store clean cache
    clean_key_cache = key_cache.clone()
    clean_value_cache = value_cache.clone()

    hamming_stats = torch.zeros(4, dtype=torch.int64, device='cuda')

    # Decode without errors to get baseline RMSE (quantization error only)
    decoded_k = torch.empty(num_tokens, num_heads, head_size, dtype=torch.float16, device='cuda')
    decoded_v = torch.empty_like(decoded_k)

    blocks_needed = (num_tokens + block_size - 1) // block_size
    block_table = torch.arange(blocks_needed, dtype=torch.int32, device='cuda').unsqueeze(0)
    cu_seq_lens = torch.tensor([0, num_tokens], dtype=torch.int32, device='cuda')
    seq_starts = torch.tensor([0], dtype=torch.int32, device='cuda')

    cp_gather_and_ecc_decode_kv_cache(
        clean_key_cache, clean_value_cache,
        decoded_k, decoded_v,
        block_table, cu_seq_lens, seq_starts,
        k_scale, v_scale, kv_cache_dtype,
        None, None, hamming_stats, None
    )
    torch.cuda.synchronize()

    baseline_k_rmse = torch.sqrt(((decoded_k - original_k) ** 2).mean()).item()
    baseline_v_rmse = torch.sqrt(((decoded_v - original_v) ** 2).mean()).item()

    results["baseline"] = {
        "ber": 0,
        "k_rmse": baseline_k_rmse,
        "v_rmse": baseline_v_rmse,
        "bits_flipped": 0,
        "hamming_no_error": 0,
        "hamming_corrected": 0,
        "hamming_detected_uncorrectable": 0,
    }

    # Test each BER level
    for ber in ber_levels:
        if ber == 0:
            continue

        ber_key = f"ber_{ber:.0e}"

        # Reset cache to clean state
        key_cache.copy_(clean_key_cache)
        value_cache.copy_(clean_value_cache)
        hamming_stats.zero_()

        # Count bits before
        k_before = key_cache.clone()
        v_before = value_cache.clone()

        # Inject errors
        inject_cache_errors(key_cache, value_cache, ber, seed)
        torch.cuda.synchronize()

        # Count actual bits flipped
        k_xor = (key_cache ^ k_before).flatten().cpu()
        v_xor = (value_cache ^ v_before).flatten().cpu()
        bits_flipped = 0
        for byte in k_xor.tolist():
            bits_flipped += bin(byte).count('1')
        for byte in v_xor.tolist():
            bits_flipped += bin(byte).count('1')

        # Decode with ECC correction
        cp_gather_and_ecc_decode_kv_cache(
            key_cache, value_cache,
            decoded_k, decoded_v,
            block_table, cu_seq_lens, seq_starts,
            k_scale, v_scale, kv_cache_dtype,
            None, None, hamming_stats, None
        )
        torch.cuda.synchronize()

        k_rmse = torch.sqrt(((decoded_k - original_k) ** 2).mean()).item()
        v_rmse = torch.sqrt(((decoded_v - original_v) ** 2).mean()).item()

        stats = hamming_stats.cpu().tolist()
        total_bits = key_cache.numel() * 8 * 2

        results[ber_key] = {
            "ber": ber,
            "bits_flipped": bits_flipped,
            "total_bits": total_bits,
            "measured_ber": bits_flipped / total_bits if total_bits > 0 else 0,
            "k_rmse": k_rmse,
            "v_rmse": v_rmse,
            "rmse_degradation_pct": ((k_rmse + v_rmse) / (baseline_k_rmse + baseline_v_rmse) - 1) * 100,
            "hamming_no_error": stats[0],
            "hamming_corrected": stats[1],
            "hamming_detected_uncorrectable": stats[2],
            "hamming_parity_only": stats[3],
            "correction_rate": stats[1] / (stats[1] + stats[2]) if (stats[1] + stats[2]) > 0 else 1.0,
        }

    return results


# ============================================================================
# GPU Utilities
# ============================================================================

def cleanup_gpu():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
    }


# ============================================================================
# Main Evaluation Functions
# ============================================================================

@dataclass
class EvalResult:
    """Single evaluation result."""
    cache_mode: str
    ber: float
    seed: int
    perplexity_mean: float
    perplexity_std: float
    perplexity_median: float
    perplexity_ci_lower: float
    perplexity_ci_upper: float
    catastrophic_rate: float
    n_samples: int
    n_valid: int
    elapsed_seconds: float
    per_sample_ppls: List[float] = field(default_factory=list)


def evaluate_single_config(
    model_name: str,
    cache_mode: str,
    texts: List[str],
    ber: float,
    seed: int,
    config: EvalConfig,
) -> EvalResult:
    """Evaluate a single configuration (model + cache_mode + BER + seed)."""
    from vllm import LLM

    start_time = time.time()

    kv_cache_dtype = CACHE_MODE_TO_VLLM_DTYPE.get(cache_mode, "auto")

    print(f"    Loading model with kv_cache_dtype={kv_cache_dtype}...")

    llm = LLM(
        model=model_name,
        kv_cache_dtype=kv_cache_dtype,
        dtype='float16',
        enforce_eager=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_length + 64,
    )

    print(f"    Computing perplexity...")
    mean_ppl, per_sample_ppls = compute_perplexity_vllm(llm, texts, config.max_length)

    # Compute statistics
    valid_ppls = [p for p in per_sample_ppls if not math.isinf(p)]
    mean, std, ci_lower, ci_upper = compute_confidence_interval(valid_ppls)
    median = float(np.median(valid_ppls)) if valid_ppls else float("inf")
    catastrophic = compute_catastrophic_rate(per_sample_ppls, config.catastrophic_threshold)

    elapsed = time.time() - start_time

    del llm
    cleanup_gpu()

    return EvalResult(
        cache_mode=cache_mode,
        ber=ber,
        seed=seed,
        perplexity_mean=mean,
        perplexity_std=std,
        perplexity_median=median,
        perplexity_ci_lower=ci_lower,
        perplexity_ci_upper=ci_upper,
        catastrophic_rate=catastrophic,
        n_samples=len(per_sample_ppls),
        n_valid=len(valid_ppls),
        elapsed_seconds=elapsed,
        per_sample_ppls=per_sample_ppls,
    )


def run_perplexity_evaluation(
    config: EvalConfig,
    texts: List[str],
) -> List[EvalResult]:
    """Run perplexity evaluation for all cache modes at BER=0 (baseline)."""
    results = []
    total_configs = len(config.cache_modes) * len(config.seeds)
    current = 0

    for cache_mode in config.cache_modes:
        print(f"\n{'='*60}")
        print(f"Cache Mode: {CACHE_MODE_LABELS.get(cache_mode, cache_mode)}")
        print(f"{'='*60}")

        for seed in config.seeds:
            current += 1
            print(f"\n  [{current}/{total_configs}] {cache_mode}, seed={seed}")

            try:
                result = evaluate_single_config(
                    config.model,
                    cache_mode,
                    texts,
                    ber=0,
                    seed=seed,
                    config=config,
                )
                results.append(result)

                print(f"    PPL: {result.perplexity_mean:.2f} +/- {result.perplexity_std:.2f}")
                print(f"    Catastrophic: {result.catastrophic_rate*100:.1f}%")
                print(f"    Time: {result.elapsed_seconds:.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                cleanup_gpu()

    return results


def run_perplexity_with_fault_injection(
    config: EvalConfig,
    texts: List[str],
) -> Dict[str, Dict[float, Dict[str, Any]]]:
    """Run perplexity evaluation under fault injection for all cache modes and BER levels.

    Returns:
        Dict mapping cache_mode -> ber -> {perplexity metrics}
    """
    results = {}

    # Skip FP16 for now as it requires different injection approach
    injectable_modes = [m for m in config.cache_modes if m != "fp16"]

    total = len(injectable_modes) * len(config.ber_levels)
    current = 0

    for cache_mode in injectable_modes:
        print(f"\n{'='*60}")
        print(f"Fault Injection PPL: {CACHE_MODE_LABELS.get(cache_mode, cache_mode)}")
        print(f"{'='*60}")

        results[cache_mode] = {}

        for ber in config.ber_levels:
            current += 1
            print(f"\n  [{current}/{total}] {cache_mode}, BER={ber:.0e}")

            try:
                from vllm import LLM

                kv_cache_dtype = CACHE_MODE_TO_VLLM_DTYPE.get(cache_mode, "auto")

                # Enable prefix caching to allow KV cache reuse after fault injection
                llm = LLM(
                    model=config.model,
                    kv_cache_dtype=kv_cache_dtype,
                    dtype='float16',
                    enforce_eager=True,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    max_model_len=config.max_length + 64,
                    enable_prefix_caching=True,  # Required for fault injection testing
                )

                mean_ppl, per_sample_ppls, stats = compute_perplexity_with_fault_injection(
                    llm,
                    texts,
                    ber=ber,
                    seed=config.seeds[0] if config.seeds else 42,
                    max_length=config.max_length,
                )

                valid_ppls = [p for p in per_sample_ppls if not math.isinf(p)]
                mean, std, ci_lower, ci_upper = compute_confidence_interval(valid_ppls)
                catastrophic = compute_catastrophic_rate(per_sample_ppls, config.catastrophic_threshold)

                results[cache_mode][ber] = {
                    "perplexity_mean": mean,
                    "perplexity_std": std,
                    "perplexity_ci_lower": ci_lower,
                    "perplexity_ci_upper": ci_upper,
                    "catastrophic_rate": catastrophic,
                    "n_samples": len(per_sample_ppls),
                    "n_valid": len(valid_ppls),
                    "stats": stats,
                }

                print(f"    PPL: {mean:.2f} +/- {std:.2f}")
                print(f"    Catastrophic: {catastrophic*100:.1f}%")

                del llm
                cleanup_gpu()

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                results[cache_mode][ber] = {"error": str(e)}
                cleanup_gpu()

    return results


def run_kernel_fault_injection_tests(
    cache_modes: List[str],
    ber_levels: List[float],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run kernel-level fault injection tests for all cache modes."""
    print("\n" + "="*60)
    print("KERNEL-LEVEL FAULT INJECTION TESTS")
    print("="*60)
    print("Testing error resilience under various bit error rates...")
    print("(FP16 has no protection, ECC modes correct errors)")

    results = {}

    for cache_mode in cache_modes:
        print(f"\n  Testing {CACHE_MODE_LABELS.get(cache_mode, cache_mode)}...")

        # Special handling for FP16 - test degradation without ECC
        if cache_mode == "fp16":
            try:
                result = test_fp16_fault_injection(
                    ber_levels=ber_levels,
                    seed=seed,
                )
                results[cache_mode] = result

                # Print summary
                print(f"    Baseline RMSE: 0.0000 (no quantization)")
                for ber in ber_levels:
                    if ber == 0:
                        continue
                    ber_key = f"ber_{ber:.0e}"
                    if ber_key in result:
                        r = result[ber_key]
                        print(f"    BER={ber:.0e}: flipped={r['bits_flipped']}, "
                              f"RMSE={r['k_rmse']:.4f} (NO CORRECTION)")
            except Exception as e:
                print(f"    Error: {e}")
                results[cache_mode] = {"error": str(e)}

            cleanup_gpu()
            continue

        try:
            result = test_fault_injection_kernel(
                cache_mode=cache_mode,
                ber_levels=ber_levels,
                seed=seed,
            )
            results[cache_mode] = result

            # Print summary
            if "baseline" in result:
                print(f"    Baseline RMSE: {result['baseline']['k_rmse']:.4f}")
            for ber in ber_levels:
                if ber == 0:
                    continue
                ber_key = f"ber_{ber:.0e}"
                if ber_key in result:
                    r = result[ber_key]
                    print(f"    BER={ber:.0e}: flipped={r['bits_flipped']}, "
                          f"corrected={r['hamming_corrected']}, "
                          f"RMSE={r['k_rmse']:.4f}")

        except Exception as e:
            print(f"    Error: {e}")
            results[cache_mode] = {"error": str(e)}

        cleanup_gpu()

    return results


def run_throughput_tests(
    cache_modes: List[str],
    iterations: int = 100,
) -> Dict[str, Any]:
    """Run throughput tests for all cache modes."""
    print("\n" + "="*60)
    print("THROUGHPUT MEASUREMENT")
    print("="*60)

    results = {}

    for cache_mode in cache_modes:
        print(f"\n  Measuring {CACHE_MODE_LABELS.get(cache_mode, cache_mode)}...")

        try:
            result = measure_kernel_throughput(
                cache_mode=cache_mode,
                iterations=iterations,
            )
            results[cache_mode] = result

            if "encode_throughput_mvalues_sec" in result:
                print(f"    Encode: {result['encode_throughput_mvalues_sec']:.1f} M values/sec")
                print(f"    Decode: {result['decode_throughput_mvalues_sec']:.1f} M values/sec")

        except Exception as e:
            print(f"    Error: {e}")
            results[cache_mode] = {"error": str(e)}

        cleanup_gpu()

    return results


# ============================================================================
# Results Aggregation and Formatting
# ============================================================================

def aggregate_perplexity_results(results: List[EvalResult]) -> Dict[str, Any]:
    """Aggregate perplexity results across seeds."""
    aggregated = {}

    # Group by cache_mode
    groups = {}
    for r in results:
        if r.cache_mode not in groups:
            groups[r.cache_mode] = []
        groups[r.cache_mode].append(r)

    for cache_mode, group in groups.items():
        ppls = [r.perplexity_mean for r in group if not math.isinf(r.perplexity_mean)]
        cats = [r.catastrophic_rate for r in group]

        if ppls:
            mean, std, ci_lower, ci_upper = compute_confidence_interval(ppls)
        else:
            mean, std, ci_lower, ci_upper = float("inf"), 0.0, float("inf"), float("inf")

        aggregated[cache_mode] = {
            "n_seeds": len(group),
            "perplexity_mean": mean,
            "perplexity_std": std,
            "perplexity_ci_lower": ci_lower,
            "perplexity_ci_upper": ci_upper,
            "catastrophic_rate_mean": float(np.mean(cats)) if cats else 0.0,
            "catastrophic_rate_std": float(np.std(cats)) if len(cats) > 1 else 0.0,
        }

    return aggregated


def print_perplexity_table(aggregated: Dict[str, Any], fp16_ppl: float = None):
    """Print perplexity comparison table."""
    print("\n" + "="*70)
    print("PERPLEXITY COMPARISON TABLE")
    print("="*70)
    print(f"{'Cache Mode':<25} {'PPL':>10} {'Std':>8} {'95% CI':>18} {'Degrad.':>10}")
    print("-"*70)

    if fp16_ppl is None and "fp16" in aggregated:
        fp16_ppl = aggregated["fp16"]["perplexity_mean"]

    for mode in ALL_CACHE_MODES:
        if mode not in aggregated:
            continue
        data = aggregated[mode]
        ppl = data["perplexity_mean"]
        std = data["perplexity_std"]
        ci_l = data["perplexity_ci_lower"]
        ci_u = data["perplexity_ci_upper"]

        if math.isinf(ppl):
            ppl_str = "inf"
            ci_str = "N/A"
            deg_str = "N/A"
        else:
            ppl_str = f"{ppl:.2f}"
            ci_str = f"[{ci_l:.2f}, {ci_u:.2f}]"
            if fp16_ppl and not math.isinf(fp16_ppl) and fp16_ppl > 0:
                degradation = (ppl - fp16_ppl) / fp16_ppl * 100
                deg_str = f"+{degradation:.1f}%"
            else:
                deg_str = "-"

        label = CACHE_MODE_LABELS.get(mode, mode)
        print(f"{label:<25} {ppl_str:>10} {std:>8.2f} {ci_str:>18} {deg_str:>10}")


def print_memory_table(memory_metrics: Dict[str, Dict]):
    """Print memory efficiency table."""
    print("\n" + "="*70)
    print("MEMORY EFFICIENCY TABLE")
    print("="*70)
    print(f"{'Cache Mode':<25} {'Bits/Val':>10} {'Compress':>10} {'ECC Over.':>12}")
    print("-"*70)

    for mode in ALL_CACHE_MODES:
        if mode not in memory_metrics:
            continue
        m = memory_metrics[mode]
        label = CACHE_MODE_LABELS.get(mode, mode)
        print(f"{label:<25} {m['bits_per_value']:>10} {m['compression_ratio']:>10.2f}x "
              f"{m['memory_overhead_pct']:>11.1f}%")


def print_fault_injection_table(fi_results: Dict[str, Any], ber_levels: List[float]):
    """Print fault injection results table."""
    print("\n" + "="*80)
    print("FAULT INJECTION ERROR RESILIENCE TABLE")
    print("="*80)
    print("Comparing FP16 (no protection) vs ECC-protected modes under bit errors")

    # Print RMSE comparison (most important metric)
    print("\nReconstruction RMSE (lower is better):")
    print("-"*80)
    header = f"{'Cache Mode':<25} {'Baseline':>10}"
    for ber in ber_levels:
        if ber > 0:
            header += f" BER={ber:.0e}"
    print(header)
    print("-"*80)

    for mode in ALL_CACHE_MODES:
        if mode not in fi_results:
            continue
        result = fi_results[mode]
        if "error" in result:
            continue

        label = CACHE_MODE_LABELS.get(mode, mode)
        baseline_rmse = result.get("baseline", {}).get("k_rmse", 0)
        row = f"{label:<25} {baseline_rmse:>10.4f}"

        for ber in ber_levels:
            if ber == 0:
                continue
            ber_key = f"ber_{ber:.0e}"
            if ber_key in result:
                rmse = result[ber_key].get("k_rmse", 0)
                row += f" {rmse:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Print correction rates (for ECC modes only)
    print("\nError Correction Rate (% of errors corrected):")
    print("-"*80)
    header = f"{'Cache Mode':<25}"
    for ber in ber_levels:
        if ber > 0:
            header += f" BER={ber:.0e}"
    print(header)
    print("-"*80)

    for mode in ALL_CACHE_MODES:
        if mode not in fi_results:
            continue
        result = fi_results[mode]
        if "error" in result:
            continue

        label = CACHE_MODE_LABELS.get(mode, mode)
        row = f"{label:<25}"

        for ber in ber_levels:
            if ber == 0:
                continue
            ber_key = f"ber_{ber:.0e}"
            if ber_key in result:
                rate = result[ber_key].get("correction_rate", 0) * 100
                if mode == "fp16" or mode == "int4":
                    row += f" {'0.0% (none)':>10}"
                else:
                    row += f" {rate:>9.1f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)


def print_throughput_table(throughput_results: Dict[str, Any]):
    """Print throughput results table."""
    print("\n" + "="*70)
    print("THROUGHPUT TABLE")
    print("="*70)
    print(f"{'Cache Mode':<25} {'Encode (M/s)':>15} {'Decode (M/s)':>15}")
    print("-"*70)

    for mode in ALL_CACHE_MODES:
        if mode not in throughput_results:
            continue
        r = throughput_results[mode]
        label = CACHE_MODE_LABELS.get(mode, mode)

        if "error" in r or "skipped" in r:
            print(f"{label:<25} {'N/A':>15} {'N/A':>15}")
        else:
            enc = r.get("encode_throughput_mvalues_sec", 0)
            dec = r.get("decode_throughput_mvalues_sec", 0)
            print(f"{label:<25} {enc:>15.1f} {dec:>15.1f}")


def generate_latex_tables(output: Dict[str, Any]) -> str:
    """Generate LaTeX tables for ICML paper."""
    latex = []

    # Perplexity table
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Perplexity comparison of ECC-protected KV cache modes}")
    latex.append(r"\begin{tabular}{lcccc}")
    latex.append(r"\toprule")
    latex.append(r"Cache Mode & PPL & $\pm$ Std & 95\% CI & Degradation \\")
    latex.append(r"\midrule")

    aggregated = output.get("aggregated_perplexity", {})
    fp16_ppl = aggregated.get("fp16", {}).get("perplexity_mean", 0)

    for mode in ALL_CACHE_MODES:
        if mode not in aggregated:
            continue
        data = aggregated[mode]
        ppl = data["perplexity_mean"]
        std = data["perplexity_std"]
        ci_l = data["perplexity_ci_lower"]
        ci_u = data["perplexity_ci_upper"]

        if not math.isinf(ppl):
            if fp16_ppl and not math.isinf(fp16_ppl) and fp16_ppl > 0:
                deg = (ppl - fp16_ppl) / fp16_ppl * 100
                deg_str = f"+{deg:.1f}\\%"
            else:
                deg_str = "-"

            label = CACHE_MODE_LABELS.get(mode, mode).replace("(", "\\texttt{(").replace(")", ")}")
            latex.append(f"{label} & {ppl:.2f} & {std:.2f} & [{ci_l:.2f}, {ci_u:.2f}] & {deg_str} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


# ============================================================================
# Main Entry Point
# ============================================================================

def print_perplexity_fault_injection_table(
    fi_ppl_results: Dict[str, Dict[float, Dict[str, Any]]],
    ber_levels: List[float]
):
    """Print perplexity under fault injection table."""
    print("\n" + "="*80)
    print("PERPLEXITY UNDER FAULT INJECTION (KEY ICML RESULT)")
    print("="*80)
    print("Shows how perplexity degrades under bit errors for each cache mode")
    print("-"*80)

    # Header
    header = f"{'Cache Mode':<25}"
    for ber in ber_levels:
        header += f" BER={ber:.0e}"
    print(header)
    print("-"*80)

    for mode in ALL_CACHE_MODES:
        if mode not in fi_ppl_results:
            continue

        label = CACHE_MODE_LABELS.get(mode, mode)
        row = f"{label:<25}"

        for ber in ber_levels:
            if ber in fi_ppl_results[mode]:
                data = fi_ppl_results[mode][ber]
                if "error" in data:
                    row += f" {'ERROR':>10}"
                else:
                    ppl = data.get("perplexity_mean", float("inf"))
                    if math.isinf(ppl):
                        row += f" {'inf':>10}"
                    else:
                        row += f" {ppl:>10.2f}"
            else:
                row += f" {'N/A':>10}"

        print(row)

    # Print degradation from baseline
    print("\nPerplexity Degradation (% increase from BER=0):")
    print("-"*80)

    for mode in ALL_CACHE_MODES:
        if mode not in fi_ppl_results:
            continue

        label = CACHE_MODE_LABELS.get(mode, mode)
        row = f"{label:<25}"

        baseline_ppl = None
        if 0 in fi_ppl_results[mode]:
            baseline_ppl = fi_ppl_results[mode][0].get("perplexity_mean")

        for ber in ber_levels:
            if ber in fi_ppl_results[mode]:
                data = fi_ppl_results[mode][ber]
                if "error" in data:
                    row += f" {'ERROR':>10}"
                else:
                    ppl = data.get("perplexity_mean", float("inf"))
                    if baseline_ppl and not math.isinf(ppl) and not math.isinf(baseline_ppl):
                        deg = (ppl - baseline_ppl) / baseline_ppl * 100
                        row += f" {deg:>+9.1f}%"
                    else:
                        row += f" {'N/A':>10}"
            else:
                row += f" {'N/A':>10}"

        print(row)


def run_comprehensive_evaluation(config: EvalConfig = None) -> Dict[str, Any]:
    """Run comprehensive ICML evaluation."""
    if config is None:
        config = EvalConfig()

    print("="*70)
    print("ECC-PROTECTED KV CACHE COMPREHENSIVE EVALUATION")
    print("="*70)
    print(f"Model: {config.model}")
    print(f"Dataset: {config.dataset}")
    print(f"Max samples: {config.max_samples}")
    print(f"Max length: {config.max_length}")
    print(f"Cache modes: {config.cache_modes}")
    print(f"BER levels: {config.ber_levels}")
    print(f"Seeds: {config.seeds}")
    print(f"GPU: {get_gpu_info().get('device_name', 'N/A')}")
    print("="*70)

    # Load evaluation texts
    print("\n[1/5] Loading evaluation texts...")
    texts = load_evaluation_texts(config.dataset, config.max_samples)
    print(f"Loaded {len(texts)} texts")

    # Run baseline perplexity evaluation (BER=0)
    print("\n[2/5] Running baseline perplexity evaluation (BER=0)...")
    ppl_results = run_perplexity_evaluation(config, texts)
    aggregated_ppl = aggregate_perplexity_results(ppl_results)

    # Run memory efficiency analysis
    print("\n[3/5] Computing memory efficiency metrics...")
    memory_metrics = compute_memory_metrics(config.cache_modes)

    # Run kernel-level fault injection tests (RMSE) - demonstrates ECC effectiveness
    print("\n[4/5] Running kernel-level fault injection tests...")
    fi_results = run_kernel_fault_injection_tests(
        config.cache_modes,
        config.ber_levels,
        seed=config.seeds[0] if config.seeds else 42,
    )

    # Run throughput tests
    print("\n[5/5] Measuring kernel throughput...")
    throughput_results = run_throughput_tests(
        config.cache_modes,
        iterations=config.throughput_iterations,
    )

    # Build output
    output = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "gpu_info": get_gpu_info(),
        "n_texts": len(texts),
        "perplexity_results": [asdict(r) for r in ppl_results],
        "aggregated_perplexity": aggregated_ppl,
        "memory_metrics": memory_metrics,
        "fault_injection_results": fi_results,  # Kernel-level RMSE under BER
        "throughput_results": throughput_results,
        "ecc_capabilities": ECC_CAPABILITIES,
    }

    # Print summary tables
    print_perplexity_table(aggregated_ppl)
    print_memory_table(memory_metrics)
    print_fault_injection_table(fi_results, config.ber_levels)  # RMSE under fault injection
    print_throughput_table(throughput_results)

    # Save results
    output_path = config.output_file
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Generate LaTeX tables
    latex_tables = generate_latex_tables(output)
    latex_path = output_path.replace('.json', '_tables.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_tables)
    print(f"LaTeX tables saved to: {latex_path}")

    return output


def main():
    """Main entry point - runs comprehensive evaluation automatically."""
    # Default ICML-quality configuration
    config = EvalConfig(
        model="gpt2",
        dataset="wikitext2",
        max_samples=50,
        max_length=256,
        stride=128,
        seeds=[42],
        ber_levels=[0, 1e-5, 1e-4, 1e-3, 1e-2],
        cache_modes=ALL_CACHE_MODES,
        gpu_memory_utilization=0.5,
        output_file="icml_eval_results.json",
        throughput_iterations=100,
    )

    try:
        output = run_comprehensive_evaluation(config)

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)

        # Print key findings
        agg = output.get("aggregated_perplexity", {})
        if "fp16" in agg and "int4-hamming84" in agg:
            fp16_ppl = agg["fp16"]["perplexity_mean"]
            h84_ppl = agg["int4-hamming84"]["perplexity_mean"]
            if not math.isinf(fp16_ppl) and not math.isinf(h84_ppl):
                deg = (h84_ppl - fp16_ppl) / fp16_ppl * 100
                print(f"\nKey Finding: Hamming(8,4) SECDED achieves {h84_ppl:.2f} PPL")
                print(f"             ({deg:+.1f}% vs FP16 baseline of {fp16_ppl:.2f})")
                print(f"             with 2x memory compression and 1-bit error correction")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
