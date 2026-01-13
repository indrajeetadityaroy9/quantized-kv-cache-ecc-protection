"""
ECC-Protected KV Cache Shim: Pluggable error-correcting codes for transformer attention.

This module implements a paged KV cache with pluggable ECC codecs (Hamming, Golay) for
bit-error resilience during inference. It replaces the model's native attention with
an ECC-protected version that quantizes K,V to INT4, applies ECC encoding, and optionally
injects bit errors for BER sweep experiments.

Architecture:
    1. ECCShimConfig: Configuration for codec, BER, and interpolation settings
    2. SimpleBlockManager: Paged memory manager for K,V cache blocks
    3. ECCBackend: Core encode/decode logic with fault injection
    4. ECCPagedAttentionShim: Drop-in replacement for model attention layers
    5. patch_model_with_ecc_attention: Context manager to wrap any HuggingFace model

Supported Codecs:
    - fp16: Unprotected FP16 oracle baseline
    - fp8: Unprotected FP8 E4M3 (vLLM standard)
    - int4: Unprotected INT4 (BER injection without ECC)
    - hamming74: Hamming(7,4) SEC (single error correct)
    - hamming84: Hamming(8,4) SECDED (single error correct, double error detect)
    - golay: Golay(24,12) (corrects up to 3 errors per 24-bit codeword)

Fault Injection Strategy:
    seed_used = config.seed + _injection_count
    where _injection_count increments per (K, V) write operation.
    Determinism guarantee: Fixed (config.seed, write_order) → identical bit flips.

Usage:
    config = ECCShimConfig(codec="hamming84", ber=1e-4, inject_errors=True)
    with patch_model_with_ecc_attention(model, config) as patched_model:
        outputs = patched_model.generate(...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import math

from kv_cache.paged_cache_ecc import compute_quantization_scales
from ecc_codecs.triton_kernels import (
    hamming74_encode,
    hamming74_decode,
    hamming84_encode,
    hamming84_decode,
    golay_encode,
    golay_decode,
    inject_bit_errors_triton,
    interpolate_double_errors,
)
from kv_cache.attention_ecc import paged_attention_ecc

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


def compute_injection_seed(base_seed: int, layer_idx: int, injection_count: int) -> int:
    """
    Compute deterministic seed for fault injection.

    Formula: base_seed + layer_idx * 10000 + injection_count

    This ensures:
    - Different layers get different random sequences
    - Different injection operations get different sequences
    - Reproducibility: same (base_seed, layer_idx, injection_count) → same result

    Supports:
    - Up to 10000 injections per layer before potential collision
    - Up to 2^31 / 10000 ≈ 214000 layers

    Args:
        base_seed: Base seed from configuration
        layer_idx: Layer index (0-based)
        injection_count: Number of injections so far for this component

    Returns:
        Deterministic seed for RNG
    """
    return base_seed + layer_idx * 10000 + injection_count


class ECCDummyCache:
    """Dummy cache that satisfies the transformers cache interface.

    The ECC shim manages its own KV cache internally, so this is just
    a placeholder to prevent errors when use_cache=True.
    """

    def __init__(self, num_layers=0):
        self.key_cache = []
        self.value_cache = []
        self._num_layers = num_layers
        self._seen_tokens = 0

    def __len__(self):
        """Return number of layers in the cache."""
        return self._num_layers

    def __iter__(self):
        """Iterate over cache layers."""
        for i in range(len(self.key_cache)):
            yield (self.key_cache[i], self.value_cache[i])

    def __getitem__(self, layer_idx):
        """Get cache for a specific layer."""
        if layer_idx < len(self.key_cache):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        return (None, None)

    def to_legacy_cache(self):
        """Return empty tuple for legacy cache format."""
        return ()

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    def get_max_length(self):
        return None

    def get_usable_length(self, new_seq_length, layer_idx=0):
        """Return usable cache length."""
        return self._seen_tokens

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._seen_tokens += key_states.shape[-2]
        return key_states, value_states

    @property
    def seen_tokens(self):
        return self._seen_tokens


class ECCShimConfig:
    """
    Configuration for ECC-protected KV cache.

    Controls codec selection, error injection, and recovery strategies for
    BER sweep experiments.

    Attributes:
        codec: ECC codec name. Options:
            - "fp16": Unprotected FP16 baseline (oracle)
            - "fp8": Unprotected FP8 E4M3 (vLLM standard)
            - "int4": Unprotected INT4 (for BER comparison without ECC)
            - "hamming74": Hamming(7,4) SEC - corrects 1 error
            - "hamming84": Hamming(8,4) SECDED - corrects 1, detects 2
            - "golay": Golay(24,12) - corrects up to 3 errors
        ber: Bit Error Rate for fault injection (0.0 = no errors)
        block_size: Tokens per cache block (default 16)
        num_blocks: Maximum number of cache blocks to allocate
        inject_errors: If True, inject bit errors during cache writes
        seed: Base RNG seed for deterministic fault injection
        use_interpolation: If True, apply linear interpolation for double errors

    Seed Strategy:
        actual_seed = seed + injection_count
        This ensures different K,V pairs get different random bit flips while
        remaining deterministic across runs with the same seed.
    """

    # Supported codecs for ECC protection
    SUPPORTED_CODECS = {"fp16", "fp8", "int4", "hamming74", "hamming84", "golay"}

    def __init__(
        self,
        codec="hamming84",
        ber=0.0,
        block_size=16,
        num_blocks=256,
        inject_errors=False,
        seed=42,
        use_interpolation=False,
    ):
        if codec not in self.SUPPORTED_CODECS:
            raise ValueError(
                f"Unsupported codec: '{codec}'. "
                f"Supported codecs: {sorted(self.SUPPORTED_CODECS)}"
            )
        self.codec = codec
        self.ber = ber
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.inject_errors = inject_errors
        self.seed = seed
        self.use_interpolation = use_interpolation


class SimpleBlockManager:
    """
    Paged memory manager for ECC-protected KV cache.

    Manages block allocation and provides storage tensors for K,V cache with
    codec-appropriate layouts. Each block stores `block_size` tokens worth of
    K or V data for all layers and KV heads.

    Memory Layout:
        k_cache, v_cache: [num_blocks, num_layers, num_kv_heads, codewords_per_head]
            - codewords_per_head = block_size * head_dim for per-element codecs
            - codewords_per_head = block_size * (head_dim/3) for Golay (triplet packing)
        k_scales, v_scales: [num_blocks, num_layers, num_kv_heads, block_size]
            - Per-token quantization scale factors

    Block Table:
        Maps (sequence_id, logical_block_idx) → physical_block_idx
        Allows non-contiguous allocation for dynamic sequence lengths.

    Supported Codecs and Their Storage:
        - fp16: torch.float16, no scales needed
        - fp8: torch.float8_e4m3fn, no scales needed
        - int4/hamming74/hamming84: torch.uint8, scales required
        - golay: torch.int32 (24-bit codewords), scales required

    Attributes:
        k_cache, v_cache: Main KV storage tensors
        k_scales, v_scales: Quantization scales (only for quantized codecs)
        block_table: [max_seqs, max_blocks_per_seq] mapping logical → physical
        free_blocks: List of unallocated block indices
        seq_to_blocks: Dict mapping sequence_id → list of physical blocks
    """

    def __init__(
        self,
        num_blocks,
        block_size,
        num_layers,
        num_kv_heads,
        head_dim,
        device="cuda",
        codec="hamming84",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.codec = codec

        if codec == "fp16":
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.float16
            self.needs_scales = False
        elif codec == "fp8":
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.float8_e4m3fn
            self.needs_scales = False
        elif codec == "int4":
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True
        elif codec == "hamming74":
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True
        elif codec == "hamming84":
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True
        elif codec == "golay":
            self.values_per_head = block_size * ((head_dim + 2) // 3)
            self.cache_dtype = torch.int32
            self.needs_scales = True
        else:
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True

        self.codewords_per_head = self.values_per_head

        self.k_cache = torch.zeros(
            num_blocks,
            num_layers,
            num_kv_heads,
            self.codewords_per_head,
            dtype=self.cache_dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            num_blocks,
            num_layers,
            num_kv_heads,
            self.codewords_per_head,
            dtype=self.cache_dtype,
            device=device,
        )

        self.k_scales = torch.zeros(
            num_blocks,
            num_layers,
            num_kv_heads,
            block_size,
            dtype=torch.float32,
            device=device,
        )
        self.v_scales = torch.zeros(
            num_blocks,
            num_layers,
            num_kv_heads,
            block_size,
            dtype=torch.float32,
            device=device,
        )

        self.free_blocks = list(range(num_blocks))
        self.seq_to_blocks = {}
        self.seq_to_len = {}

        self.max_seqs = 32
        self.max_blocks_per_seq = num_blocks
        self.block_table = torch.full(
            (self.max_seqs, self.max_blocks_per_seq),
            -1,
            dtype=torch.int32,
            device=device,
        )

    def allocate(self, seq_id, num_tokens):
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        existing_blocks = self.seq_to_blocks.get(seq_id, [])
        existing_len = self.seq_to_len.get(seq_id, 0)

        existing_block_count = len(existing_blocks)
        new_blocks_needed = max(0, num_blocks_needed - existing_block_count)

        if new_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Out of blocks: need {new_blocks_needed}, have {len(self.free_blocks)}"
            )

        new_blocks = [self.free_blocks.pop(0) for _ in range(new_blocks_needed)]
        all_blocks = existing_blocks + new_blocks

        self.seq_to_blocks[seq_id] = all_blocks
        self.seq_to_len[seq_id] = num_tokens

        for i, block_id in enumerate(all_blocks):
            self.block_table[seq_id, i] = block_id

        return self.block_table[seq_id], num_tokens

    def get_block_table(self, seq_id):
        return self.block_table[seq_id]

    def get_context_len(self, seq_id):
        return self.seq_to_len.get(seq_id, 0)

    def reset(self):
        for blocks in self.seq_to_blocks.values():
            self.free_blocks.extend(blocks)
        self.seq_to_blocks.clear()
        self.seq_to_len.clear()

        self.block_table.fill_(-1)

        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_scales.zero_()
        self.v_scales.zero_()


class ECCBackend:
    """
    Core ECC encode/decode engine with fault injection support.

    Implements the write() and attend() operations that bridge the model's
    attention mechanism with the ECC-protected cache. Handles:
        - Quantization: FP16 → INT4 with per-position scale factors
        - ECC Encoding: INT4 → Hamming84/Golay codewords
        - Fault Injection: Bernoulli bit flips at specified BER
        - ECC Decoding: Codewords → INT4 with error correction
        - Dequantization: INT4 → FP16 for attention computation

    Error Statistics:
        Tracks cumulative counts for analysis:
        - _errors_corrected: Single-bit errors successfully corrected (SEC)
        - _errors_detected: Double-bit errors detected but NOT corrected (DED)

        Error Type Values (from ECC decode):
            0 = NO_ERROR: No bit errors detected, original data valid
            1 = SINGLE_CORRECTED: 1-bit error corrected, data now valid
            2 = DOUBLE_DETECTED: 2-bit error detected, data CORRUPTED
                SECDED cannot correct double errors. Caller must handle:
                - Hamming84: Preserve corrupted data (default) or interpolate
                - With interpolation: Replace with average of temporal neighbors
            3 = UNCORRECTABLE: 3+ bit errors (Golay only), data is garbage

        Note: Double errors (type 2) increment _errors_detected but the
        returned data is still corrupted unless interpolation is enabled.

    Determinism:
        Fault injection uses seed = config.seed + _injection_count, where
        _injection_count is a global counter that increments per (K, V) pair
        written across all layers. This ensures:
        - Same seed + same write order = identical bit flips
        - Different K/V tensors get different random patterns
        - Different layers get different patterns (via incremental count)

        Note: FaultInjectionAttentionShim uses an alternative formula with
        explicit layer_idx * 10000 multiplier for per-layer injection counts.
        Both approaches are deterministic and valid for their use cases.

    Path Selection:
        - Triton kernel path: Used for seq_len=1 decode with hamming84
        - Python path: Used for prefill (seq_len>1) or when interpolation needed

    GQA Support:
        Handles Grouped-Query Attention by expanding K,V heads via
        repeat_interleave before attention computation.

    Attributes:
        manager: SimpleBlockManager providing cache storage
        config: ECCShimConfig with codec and BER settings
        num_heads: Number of query attention heads
        num_kv_heads: Number of K,V attention heads (may differ for GQA)
        num_kv_groups: num_heads // num_kv_heads (expansion factor for GQA)
        _injection_count: Counter for deterministic fault injection
        _errors_corrected: Cumulative single-bit errors corrected
        _errors_detected: Cumulative double-bit errors detected
    """

    def __init__(
        self,
        manager,
        config,
        num_heads,
    ):
        self.manager = manager
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = manager.num_kv_heads
        self.head_dim = manager.head_dim

        # GQA: multiple query heads share each K,V head
        self.num_kv_groups = num_heads // self.num_kv_heads

        # Fault injection counter for deterministic RNG
        self._injection_count = 0

        # Error statistics for analysis
        self._errors_corrected = 0  # Single-bit errors corrected
        self._errors_detected = 0  # Double-bit errors detected (SECDED)
        self._total_values = 0  # Total KV values processed

    def reset_stats(self):
        """Reset error statistics for a new text/trial.

        Called by reset_ecc_cache() to ensure fresh statistics for each text.
        Without this, error counts would leak across texts within a trial,
        contaminating reported statistics.
        """
        self._injection_count = 0
        self._errors_corrected = 0
        self._errors_detected = 0
        self._total_values = 0

    def write(
        self,
        k,
        v,
        layer_idx,
        seq_id=0,
    ):
        """
        Quantize, encode, optionally corrupt, and store K,V tensors.

        Pipeline for ECC-protected codecs:
            1. Compute per-position quantization scales: scale = max(|x|) / 7
            2. Quantize to INT4: int4_val = round(x / scale) + 8
            3. Apply ECC encoding: INT4 → codeword (Hamming84 or Golay)
            4. Inject bit errors if config.inject_errors and config.ber > 0
            5. Store codewords and scales in block manager cache

        For unprotected codecs (fp16, fp8, int4):
            - fp16: Store directly, no quantization
            - fp8: Convert to FP8 E4M3, optionally inject bit errors
            - int4: Quantize without ECC, optionally inject bit errors

        Args:
            k: Key tensor [batch, seq_len, num_kv_heads * head_dim]
            v: Value tensor [batch, seq_len, num_kv_heads * head_dim]
            layer_idx: Which transformer layer (for cache indexing)
            seq_id: Sequence identifier for block allocation

        Side Effects:
            - Allocates blocks via manager.allocate() if needed
            - Increments _injection_count for each (K, V) pair (determinism)
            - Updates manager.k_cache, v_cache, k_scales, v_scales

        Fault Injection:
            When enabled, uses seed = config.seed + _injection_count to generate
            deterministic bit flips. Separate seeds for K (+0) and V (+1).
        """
        batch_size, seq_len, hidden_kv = k.shape
        device = k.device
        head_dim = self.head_dim

        # Track total values written (K and V, all positions, all heads)
        self._total_values += 2 * batch_size * seq_len * self.num_kv_heads * head_dim

        current_len = self.manager.get_context_len(seq_id)
        if current_len < seq_len:
            self.manager.allocate(seq_id, seq_len)

        block_table = self.manager.get_block_table(seq_id)

        k_reshaped = k.view(batch_size, seq_len, self.num_kv_heads, head_dim)
        v_reshaped = v.view(batch_size, seq_len, self.num_kv_heads, head_dim)

        if self.config.codec == "fp16":
            for b in range(batch_size):
                for pos in range(seq_len):
                    logical_block = pos // self.manager.block_size
                    slot_in_block = pos % self.manager.block_size
                    physical_block = int(block_table[logical_block].item())

                    if physical_block < 0:
                        continue

                    for h in range(self.num_kv_heads):
                        k_vals = k_reshaped[b, pos, h]
                        v_vals = v_reshaped[b, pos, h]

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = k_vals
                        self.manager.v_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = v_vals
            return

        if self.config.codec == "fp8":
            for b in range(batch_size):
                for pos in range(seq_len):
                    logical_block = pos // self.manager.block_size
                    slot_in_block = pos % self.manager.block_size
                    physical_block = int(block_table[logical_block].item())

                    if physical_block < 0:
                        continue

                    for h in range(self.num_kv_heads):
                        # Convert FP16 to FP8 E4M3 format
                        k_vals = k_reshaped[b, pos, h].to(torch.float8_e4m3fn)
                        v_vals = v_reshaped[b, pos, h].to(torch.float8_e4m3fn)

                        # Inject bit errors if enabled (on raw FP8 bytes)
                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            # FP8 is 8 bits per value
                            k_vals = inject_bit_errors_triton(
                                k_vals.view(torch.uint8), self.config.ber, 8, seed
                            ).view(torch.float8_e4m3fn)
                            v_vals = inject_bit_errors_triton(
                                v_vals.view(torch.uint8), self.config.ber, 8, seed + 1
                            ).view(torch.float8_e4m3fn)

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = k_vals
                        self.manager.v_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = v_vals
            return

        k_scales = compute_quantization_scales(k_reshaped.float(), dim=-1)
        v_scales = compute_quantization_scales(v_reshaped.float(), dim=-1)

        k_int4 = (
            torch.round(k_reshaped.float() / k_scales.unsqueeze(-1)).clamp(-8, 7) + 8
        ).to(torch.uint8)
        v_int4 = (
            torch.round(v_reshaped.float() / v_scales.unsqueeze(-1)).clamp(-8, 7) + 8
        ).to(torch.uint8)

        if self.config.codec == "int4":
            for b in range(batch_size):
                for pos in range(seq_len):
                    logical_block = pos // self.manager.block_size
                    slot_in_block = pos % self.manager.block_size
                    physical_block = int(block_table[logical_block].item())

                    if physical_block < 0:
                        continue

                    for h in range(self.num_kv_heads):
                        k_vals = k_int4[b, pos, h]
                        v_vals = v_int4[b, pos, h]

                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_vals = inject_bit_errors_triton(
                                k_vals, self.config.ber, 4, seed
                            )
                            v_vals = inject_bit_errors_triton(
                                v_vals, self.config.ber, 4, seed + 1
                            )

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = k_vals
                        self.manager.v_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = v_vals

                        self.manager.k_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = k_scales[b, pos, h]
                        self.manager.v_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = v_scales[b, pos, h]
            return

        golay_padded_dim = ((head_dim + 2) // 3) * 3
        golay_num_codewords = golay_padded_dim // 3

        for b in range(batch_size):
            for pos in range(seq_len):
                logical_block = pos // self.manager.block_size
                slot_in_block = pos % self.manager.block_size
                physical_block = int(block_table[logical_block].item())

                if physical_block < 0:
                    continue

                for h in range(self.num_kv_heads):
                    k_vals = k_int4[b, pos, h]
                    v_vals = v_int4[b, pos, h]

                    if self.config.codec == "hamming74":
                        k_encoded = hamming74_encode(k_vals)
                        v_encoded = hamming74_encode(v_vals)

                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(
                                k_encoded, self.config.ber, 7, seed
                            )
                            v_encoded = inject_bit_errors_triton(
                                v_encoded, self.config.ber, 7, seed + 1
                            )

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = k_encoded
                        self.manager.v_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = v_encoded

                        self.manager.k_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = k_scales[b, pos, h]
                        self.manager.v_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = v_scales[b, pos, h]
                    elif self.config.codec == "golay":
                        k_padded = torch.zeros(
                            golay_padded_dim, dtype=torch.uint8, device=device
                        )
                        v_padded = torch.zeros(
                            golay_padded_dim, dtype=torch.uint8, device=device
                        )
                        k_padded[:head_dim] = k_vals
                        v_padded[:head_dim] = v_vals

                        k_triplets = k_padded.view(-1, 3)
                        v_triplets = v_padded.view(-1, 3)

                        k_encoded = golay_encode(k_triplets)
                        v_encoded = golay_encode(v_triplets)

                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(
                                k_encoded, self.config.ber, 24, seed
                            )
                            v_encoded = inject_bit_errors_triton(
                                v_encoded, self.config.ber, 24, seed + 1
                            )

                        offset_start = slot_in_block * golay_num_codewords
                        offset_end = offset_start + golay_num_codewords
                        self.manager.k_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = k_encoded
                        self.manager.v_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = v_encoded

                        self.manager.k_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = k_scales[b, pos, h]
                        self.manager.v_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = v_scales[b, pos, h]
                    else:
                        k_encoded = hamming84_encode(k_vals)
                        v_encoded = hamming84_encode(v_vals)

                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(
                                k_encoded, self.config.ber, 8, seed
                            )
                            v_encoded = inject_bit_errors_triton(
                                v_encoded, self.config.ber, 8, seed + 1
                            )

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = k_encoded
                        self.manager.v_cache[
                            physical_block, layer_idx, h, offset_start:offset_end
                        ] = v_encoded

                        self.manager.k_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = k_scales[b, pos, h]
                        self.manager.v_scales[
                            physical_block, layer_idx, h, slot_in_block
                        ] = v_scales[b, pos, h]

    def attend(
        self,
        q,
        layer_idx,
        seq_id=0,
    ):
        """
        Decode cache, correct errors, dequantize, and compute attention.

        Pipeline:
            1. Load encoded codewords from cache blocks
            2. Decode with ECC: codeword → INT4 + error correction
            3. Dequantize: INT4 → FP16 using stored scales
            4. (Optional) Interpolate double-error positions
            5. Expand KV heads for GQA if needed
            6. Compute scaled dot-product attention via PyTorch SDPA

        Path Selection:
            - Triton path: seq_len=1, hamming84, no interpolation
              Uses fused decode+attention kernel for ~20x speedup
            - Python path: All other cases (prefill, interpolation, Golay)
              Decodes explicitly, then uses torch.nn.functional.scaled_dot_product_attention

        Error Recovery:
            - Hamming84 without interpolation: Single errors corrected, doubles preserved
            - Hamming84 with interpolation: Double errors replaced by linear interpolation
            - Golay: Up to 3 errors corrected, ≥4 errors preserved

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Which transformer layer
            seq_id: Sequence identifier for block table lookup

        Returns:
            Output tensor [batch, num_heads, seq_len, head_dim]
            Same shape as query, containing attention-weighted sum of values.

        Side Effects:
            - Updates _errors_corrected and _errors_detected counters
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        context_len = self.manager.get_context_len(seq_id)
        if context_len == 0:
            return torch.zeros_like(q)

        block_table = self.manager.get_block_table(seq_id)
        num_ctx_blocks = (
            context_len + self.manager.block_size - 1
        ) // self.manager.block_size

        # Use Triton kernel for hamming84 decode (seq_len=1)
        # For prefill (seq_len > 1), Python path with batched SDPA is faster
        if (
            self.config.codec == "hamming84"
            and not self.config.use_interpolation
            and seq_len == 1
        ):
            return self._attend_triton_hamming84(
                q, layer_idx, seq_id, batch_size, num_heads, seq_len, head_dim, context_len
            )

        if self.config.codec == "fp16":
            k_list = []
            v_list = []

            for blk_idx in range(num_ctx_blocks):
                phys_block = int(block_table[blk_idx].item())
                if phys_block < 0:
                    continue

                start_pos = blk_idx * self.manager.block_size
                end_pos = min(start_pos + self.manager.block_size, context_len)

                for slot in range(end_pos - start_pos):
                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim

                    k_val = self.manager.k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    v_val = self.manager.v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]

                    k_list.append(k_val)
                    v_list.append(v_val)

            if not k_list:
                return torch.zeros_like(q)

            k_float = torch.stack(k_list, dim=0)
            v_float = torch.stack(v_list, dim=0)

            return self._run_attention(q, k_float, v_float, device)

        if self.config.codec == "fp8":
            k_list = []
            v_list = []

            for blk_idx in range(num_ctx_blocks):
                phys_block = int(block_table[blk_idx].item())
                if phys_block < 0:
                    continue

                start_pos = blk_idx * self.manager.block_size
                end_pos = min(start_pos + self.manager.block_size, context_len)

                for slot in range(end_pos - start_pos):
                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim

                    k_val = self.manager.k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    v_val = self.manager.v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]

                    k_list.append(k_val)
                    v_list.append(v_val)

            if not k_list:
                return torch.zeros_like(q)

            # Convert FP8 back to FP16 for attention computation
            k_fp8 = torch.stack(k_list, dim=0)
            v_fp8 = torch.stack(v_list, dim=0)
            k_float = k_fp8.to(torch.float16)
            v_float = v_fp8.to(torch.float16)

            return self._run_attention(q, k_float, v_float, device)

        if self.config.codec == "int4":
            k_list = []
            v_list = []
            k_scale_list = []
            v_scale_list = []

            for blk_idx in range(num_ctx_blocks):
                phys_block = int(block_table[blk_idx].item())
                if phys_block < 0:
                    continue

                start_pos = blk_idx * self.manager.block_size
                end_pos = min(start_pos + self.manager.block_size, context_len)

                for slot in range(end_pos - start_pos):
                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim

                    k_val = self.manager.k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    v_val = self.manager.v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                    k_list.append(k_val)
                    v_list.append(v_val)
                    k_scale_list.append(k_scale)
                    v_scale_list.append(v_scale)

            if not k_list:
                return torch.zeros_like(q)

            k_int4 = torch.stack(k_list, dim=0)
            v_int4 = torch.stack(v_list, dim=0)
            k_scales = torch.stack(k_scale_list, dim=0)
            v_scales = torch.stack(v_scale_list, dim=0)

            k_float = (k_int4.float() - 8.0) * k_scales.unsqueeze(-1)
            v_float = (v_int4.float() - 8.0) * v_scales.unsqueeze(-1)

            return self._run_attention(q, k_float, v_float, device)

        golay_padded_dim = ((head_dim + 2) // 3) * 3
        golay_num_codewords = golay_padded_dim // 3

        sink_k_enc_list = []
        sink_v_enc_list = []
        sink_k_scale_list = []
        sink_v_scale_list = []

        ctx_k_enc_list = []
        ctx_v_enc_list = []
        ctx_k_scale_list = []
        ctx_v_scale_list = []

        for blk_idx in range(num_ctx_blocks):
            phys_block = int(block_table[blk_idx].item())
            if phys_block < 0:
                continue

            start_pos = blk_idx * self.manager.block_size
            end_pos = min(start_pos + self.manager.block_size, context_len)

            is_standalone_golay = self.config.codec == "golay"

            for slot in range(end_pos - start_pos):
                global_pos = start_pos + slot

                if is_standalone_golay:
                    offset_start = slot * golay_num_codewords
                    offset_end = offset_start + golay_num_codewords

                    k_enc = self.manager.k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    v_enc = self.manager.v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                    sink_k_enc_list.append(k_enc)
                    sink_v_enc_list.append(v_enc)
                    sink_k_scale_list.append(k_scale)
                    sink_v_scale_list.append(v_scale)
                else:
                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim

                    k_enc = self.manager.k_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    v_enc = self.manager.v_cache[
                        phys_block, layer_idx, :, offset_start:offset_end
                    ]
                    k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                    ctx_k_enc_list.append(k_enc)
                    ctx_v_enc_list.append(v_enc)
                    ctx_k_scale_list.append(k_scale)
                    ctx_v_scale_list.append(v_scale)

        sink_k_float = None
        sink_v_float = None

        if sink_k_enc_list:
            sink_k_stacked = torch.stack(sink_k_enc_list, dim=0)
            sink_v_stacked = torch.stack(sink_v_enc_list, dim=0)
            sink_k_scale_stacked = torch.stack(sink_k_scale_list, dim=0)
            sink_v_scale_stacked = torch.stack(sink_v_scale_list, dim=0)

            sink_len = sink_k_stacked.shape[0]

            sink_k_dec_triplets, k_golay_stats = golay_decode(sink_k_stacked.flatten())
            sink_v_dec_triplets, v_golay_stats = golay_decode(sink_v_stacked.flatten())

            self._errors_corrected += k_golay_stats[0] + v_golay_stats[0]
            self._errors_detected += k_golay_stats[1] + v_golay_stats[1]

            total_decoded_per_pos = self.num_kv_heads * golay_padded_dim

            sink_k_dec_flat = sink_k_dec_triplets.flatten()
            sink_v_dec_flat = sink_v_dec_triplets.flatten()

            sink_k_dec = sink_k_dec_flat.view(
                sink_len, self.num_kv_heads, golay_padded_dim
            )
            sink_v_dec = sink_v_dec_flat.view(
                sink_len, self.num_kv_heads, golay_padded_dim
            )

            sink_k_dec = sink_k_dec[:, :, :head_dim]
            sink_v_dec = sink_v_dec[:, :, :head_dim]

            sink_k_float = (sink_k_dec.float() - 8.0) * sink_k_scale_stacked.unsqueeze(
                -1
            )
            sink_v_float = (sink_v_dec.float() - 8.0) * sink_v_scale_stacked.unsqueeze(
                -1
            )

        ctx_k_float = None
        ctx_v_float = None

        if ctx_k_enc_list:
            ctx_k_stacked = torch.stack(ctx_k_enc_list, dim=0)
            ctx_v_stacked = torch.stack(ctx_v_enc_list, dim=0)
            ctx_k_scale_stacked = torch.stack(ctx_k_scale_list, dim=0)
            ctx_v_scale_stacked = torch.stack(ctx_v_scale_list, dim=0)

            ctx_len_actual = ctx_k_stacked.shape[0]

            k_flat = ctx_k_stacked.flatten()
            v_flat = ctx_v_stacked.flatten()

            if self.config.codec == "hamming74":
                ctx_k_dec, k_h74_stats = hamming74_decode(k_flat)
                ctx_v_dec, v_h74_stats = hamming74_decode(v_flat)

                self._errors_corrected += k_h74_stats[0] + v_h74_stats[0]
            elif self.config.use_interpolation:
                ctx_k_dec, k_error_types, k_h84_stats = hamming84_decode(
                    k_flat, return_error_types=True
                )
                ctx_v_dec, v_error_types, v_h84_stats = hamming84_decode(
                    v_flat, return_error_types=True
                )

                self._errors_corrected += k_h84_stats[0] + v_h84_stats[0]
                self._errors_detected += k_h84_stats[1] + v_h84_stats[1]

                # Reshape for temporal interpolation along context length (not features)
                # Shape: [ctx_len * heads * dim] -> [ctx_len, heads, dim]
                # This ensures interpolation uses neighboring tokens, not adjacent features
                ctx_k_dec = ctx_k_dec.view(ctx_len_actual, self.num_kv_heads, head_dim)
                k_error_types = k_error_types.view(ctx_len_actual, self.num_kv_heads, head_dim)
                ctx_v_dec = ctx_v_dec.view(ctx_len_actual, self.num_kv_heads, head_dim)
                v_error_types = v_error_types.view(ctx_len_actual, self.num_kv_heads, head_dim)

                # Interpolate along temporal dimension (seq_dim=0 = context length)
                # Double errors at position t are replaced by average of t-1 and t+1
                ctx_k_dec = interpolate_double_errors(ctx_k_dec, k_error_types, seq_dim=0)
                ctx_v_dec = interpolate_double_errors(ctx_v_dec, v_error_types, seq_dim=0)
            else:
                ctx_k_dec, k_h84_stats = hamming84_decode(k_flat)
                ctx_v_dec, v_h84_stats = hamming84_decode(v_flat)

                self._errors_corrected += k_h84_stats[0] + v_h84_stats[0]
                self._errors_detected += k_h84_stats[1] + v_h84_stats[1]

            ctx_k_dec = ctx_k_dec.view(ctx_len_actual, self.num_kv_heads, head_dim)
            ctx_v_dec = ctx_v_dec.view(ctx_len_actual, self.num_kv_heads, head_dim)

            ctx_k_float = (ctx_k_dec.float() - 8.0) * ctx_k_scale_stacked.unsqueeze(-1)
            ctx_v_float = (ctx_v_dec.float() - 8.0) * ctx_v_scale_stacked.unsqueeze(-1)

        if sink_k_float is not None and ctx_k_float is not None:
            k_float = torch.cat(
                [sink_k_float.to(device), ctx_k_float.to(device)], dim=0
            )
            v_float = torch.cat(
                [sink_v_float.to(device), ctx_v_float.to(device)], dim=0
            )
        elif sink_k_float is not None:
            k_float = sink_k_float.to(device)
            v_float = sink_v_float.to(device)
        elif ctx_k_float is not None:
            k_float = ctx_k_float.to(device)
            v_float = ctx_v_float.to(device)
        else:
            return torch.zeros_like(q)

        return self._run_attention(q, k_float, v_float, device)

    def _attend_triton_hamming84(
        self,
        q,
        layer_idx,
        seq_id,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        context_len,
    ):
        """
        Fast path using Triton kernel for Hamming(8,4) codec (decode only).

        This provides ~20x speedup over the Python loop-based implementation
        by fusing ECC decoding directly into the attention kernel.

        Only used for seq_len=1 (decode phase). For prefill, the Python path
        with batched SDPA is faster since it avoids per-position kernel launches.
        """
        device = q.device
        block_table = self.manager.get_block_table(seq_id)

        # Decode case: single token query, call kernel directly
        q_single = q.squeeze(2)  # [batch, heads, head_dim]

        # Prepare context_lens tensor
        context_lens_tensor = torch.tensor(
            [context_len], dtype=torch.int32, device=device
        )

        output = paged_attention_ecc(
            query=q_single,
            k_cache=self.manager.k_cache,
            v_cache=self.manager.v_cache,
            block_table=block_table.unsqueeze(0),
            context_lens=context_lens_tensor,
            k_scales=self.manager.k_scales,
            layer_idx=layer_idx,
            block_size=self.manager.block_size,
            codec="hamming84",
            v_scales=self.manager.v_scales,
        )

        # Reshape back to [batch, heads, seq_len=1, head_dim]
        return output.unsqueeze(2)

    def _run_attention(
        self,
        q,
        k_float,
        v_float,
        device,
    ):
        if self.num_kv_groups > 1:
            k_float = k_float.repeat_interleave(self.num_kv_groups, dim=1)
            v_float = v_float.repeat_interleave(self.num_kv_groups, dim=1)

        k_for_sdpa = k_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
        v_for_sdpa = v_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

        # For decode (seq_len=1), don't use causal masking since the query
        # is past all cached keys. For prefill (seq_len>1), use causal masking.
        q_seq_len = q.shape[2]
        use_causal = q_seq_len > 1

        output = F.scaled_dot_product_attention(
            q,
            k_for_sdpa,
            v_for_sdpa,
            is_causal=use_causal,
        )

        return output


class ECCPagedAttentionShim(nn.Module):
    def __init__(
        self,
        original_attn,
        layer_idx,
        backend,
        rotary_emb,
        model_type="llama",
    ):
        super().__init__()
        self.model_type = model_type

        if model_type == "gpt2":
            # GPT-2 uses combined c_attn (Q+K+V) and c_proj (output)
            for proj_name in ["c_attn", "c_proj"]:
                if not hasattr(original_attn, proj_name):
                    available = [a for a in dir(original_attn) if not a.startswith("_")]
                    raise ValueError(
                        f"GPT-2 attention module missing '{proj_name}'. "
                        f"Available: {available}"
                    )
                setattr(self, proj_name, getattr(original_attn, proj_name))

            # Copy dropout layers
            if hasattr(original_attn, "attn_dropout"):
                self.attn_dropout = original_attn.attn_dropout
            if hasattr(original_attn, "resid_dropout"):
                self.resid_dropout = original_attn.resid_dropout
        else:
            # Llama/Mistral style: separate q_proj, k_proj, v_proj, o_proj
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if not hasattr(original_attn, proj_name):
                    available = [a for a in dir(original_attn) if not a.startswith("_")]
                    raise ValueError(
                        f"Attention module {type(original_attn).__name__} missing '{proj_name}'. "
                        f"Available attributes: {available}"
                    )
                proj = getattr(original_attn, proj_name)
                if proj is None:
                    raise ValueError(
                        f"'{proj_name}' is None in {type(original_attn).__name__}. "
                        f"This attention implementation may not be compatible."
                    )
                setattr(self, proj_name, proj)

        num_heads, num_kv_heads, head_dim = _get_attention_params(original_attn)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = self.num_heads * self.head_dim

        # GPT-2 may have split_size attribute
        if hasattr(original_attn, "split_size"):
            self.split_size = original_attn.split_size

        self.backend = backend
        self.layer_idx = layer_idx
        self.rotary_emb = rotary_emb

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        layer_past=None,  # GPT-2 uses this name
        head_mask=None,  # GPT-2 uses this
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        if self.model_type == "gpt2":
            return self._forward_gpt2(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past or past_key_value,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        else:
            return self._forward_llama(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

    def _forward_gpt2(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        """GPT-2 style forward pass with combined c_attn projection."""
        batch_size, seq_len, _ = hidden_states.shape

        # GPT-2 c_attn produces Q, K, V concatenated
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.split_size, dim=2)

        # Reshape to (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # No rotary embeddings for GPT-2 (uses absolute position embeddings)

        # Store K, V in ECC cache
        k_flat = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        v_flat = v.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        self.backend.write(k_flat, v_flat, self.layer_idx, seq_id=0)

        # Compute attention using ECC-protected cache
        attn_output = self.backend.attend(q, self.layer_idx, seq_id=0)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.c_proj(attn_output)

        if hasattr(self, "resid_dropout"):
            output = self.resid_dropout(output)

        # GPT-2 returns (output, present) where present is (k, v) tuple
        outputs = (output,)
        if use_cache:
            outputs = outputs + ((k, v),)
        else:
            outputs = outputs + (None,)

        if output_attentions:
            outputs = outputs + (None,)  # We don't compute attention weights

        return outputs

    def _forward_llama(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ):
        """Llama/Mistral style forward pass with separate projections."""
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        k_flat = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        v_flat = v.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        self.backend.write(k_flat, v_flat, self.layer_idx, seq_id=0)

        attn_output = self.backend.attend(q, self.layer_idx, seq_id=0)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        # Return 2 values: (output, attn_weights)
        # Newer transformers LlamaDecoderLayer expects: hidden_states, _ = self.self_attn(...)
        attn_weights = None  # We don't compute attention weights

        return output, attn_weights

    def _apply_rotary_pos_emb(
        self,
        q,
        k,
        cos,
        sin,
    ):
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        batch, q_heads, seq_len, head_dim = q.shape
        _, k_heads, _, _ = k.shape

        while cos.dim() < 4:
            cos = cos.unsqueeze(0 if cos.dim() < 2 else 1)
            sin = sin.unsqueeze(0 if sin.dim() < 2 else 1)

        if cos.shape[1] != 1 and cos.shape[1] != q_heads:
            cos = cos[:, :1, :, :]
            sin = sin[:, :1, :, :]

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


@contextmanager
def patch_model_with_ecc_attention(model, config, num_blocks=256):
    # Detect model architecture
    model_type = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama/Mistral style: model.model.layers
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens
        model_type = "llama"
    elif hasattr(model, "layers"):
        # Direct layers access
        layers = model.layers
        embed_tokens = model.embed_tokens
        model_type = "llama"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 style: model.transformer.h
        layers = model.transformer.h
        embed_tokens = model.transformer.wte
        model_type = "gpt2"
    else:
        raise ValueError("Unsupported model architecture")

    num_layers = len(layers)

    # Get first attention module based on model type
    if model_type == "gpt2":
        first_attn = layers[0].attn
    else:
        first_attn = layers[0].self_attn

    num_heads, num_kv_heads, head_dim = _get_attention_params(first_attn)

    manager = SimpleBlockManager(
        num_blocks=num_blocks,
        block_size=config.block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=next(model.parameters()).device,
        codec=config.codec,
    )

    backend = ECCBackend(manager, config, num_heads)

    original_attns = {}

    # GPT-2 uses absolute position embeddings, not rotary
    rotary_emb = None if model_type == "gpt2" else _find_rotary_embedding(model, layers)

    # Determine attribute name for attention module
    attn_attr = "attn" if model_type == "gpt2" else "self_attn"

    try:
        for layer_idx, layer in enumerate(layers):
            original_attn = getattr(layer, attn_attr)
            original_attns[layer_idx] = original_attn

            shim = ECCPagedAttentionShim(
                original_attn=original_attn,
                layer_idx=layer_idx,
                backend=backend,
                rotary_emb=rotary_emb,
                model_type=model_type,
            )

            setattr(layer, attn_attr, shim)

        model._ecc_block_manager = manager
        model._ecc_backend = backend
        model._ecc_model_type = model_type
        model._ecc_attn_attr = attn_attr

        yield model

    finally:
        attn_attr = getattr(model, "_ecc_attn_attr", "self_attn")
        for layer_idx, original_attn in original_attns.items():
            setattr(layers[layer_idx], attn_attr, original_attn)

        if hasattr(model, "_ecc_block_manager"):
            delattr(model, "_ecc_block_manager")
        if hasattr(model, "_ecc_backend"):
            delattr(model, "_ecc_backend")
        if hasattr(model, "_ecc_model_type"):
            delattr(model, "_ecc_model_type")
        if hasattr(model, "_ecc_attn_attr"):
            delattr(model, "_ecc_attn_attr")


def _find_rotary_embedding(model, layers):
    if hasattr(layers[0].self_attn, "rotary_emb"):
        return layers[0].self_attn.rotary_emb

    if hasattr(layers[0], "rotary_emb"):
        return layers[0].rotary_emb

    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb

    if hasattr(model, "rotary_emb"):
        return model.rotary_emb

    if LlamaRotaryEmbedding is not None:
        config = getattr(model, "config", None)
        if config is None and hasattr(model, "model"):
            config = getattr(model.model, "config", None)

        if config is not None:
            device = next(model.parameters()).device

            try:
                rotary_emb = LlamaRotaryEmbedding(config=config, device=device)
                return rotary_emb
            except TypeError:
                head_dim = getattr(config, "head_dim", None)
                if head_dim is None:
                    hidden_size = config.hidden_size
                    num_heads = config.num_attention_heads
                    head_dim = hidden_size // num_heads

                max_position_embeddings = getattr(
                    config, "max_position_embeddings", 4096
                )
                rope_theta = getattr(config, "rope_theta", 10000.0)

                return LlamaRotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                ).to(device)

    class SimpleRotaryEmbedding(nn.Module):
        def __init__(self, dim, base=10000.0, max_seq_len=4096):
            super().__init__()
            self.dim = dim
            self.base = base
            self.max_seq_len = max_seq_len

            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, x, position_ids):
            seq_len = position_ids.shape[-1]
            device = position_ids.device

            inv_freq = self.inv_freq.to(device)

            freqs = torch.einsum("i,j->ij", position_ids[0].float(), inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)
            sin = emb.sin().unsqueeze(0).unsqueeze(0)

            return cos, sin

    num_heads, num_kv_heads, head_dim = _get_attention_params(layers[0].self_attn)
    device = next(model.parameters()).device

    return SimpleRotaryEmbedding(dim=head_dim).to(device)


def _get_attention_params(attn_module):
    num_heads = None
    for attr in ["num_heads", "num_attention_heads"]:
        if hasattr(attn_module, attr):
            num_heads = getattr(attn_module, attr)
            break
    if num_heads is None and hasattr(attn_module, "config"):
        config = attn_module.config
        for attr in ["num_attention_heads", "num_heads", "n_head"]:
            if hasattr(config, attr):
                num_heads = getattr(config, attr)
                break

    head_dim = None
    for attr in ["head_dim", "head_size"]:
        if hasattr(attn_module, attr):
            head_dim = getattr(attn_module, attr)
            break
    if head_dim is None and hasattr(attn_module, "config"):
        config = attn_module.config
        if hasattr(config, "head_dim"):
            head_dim = config.head_dim
        elif hasattr(config, "hidden_size") and num_heads:
            head_dim = config.hidden_size // num_heads

    num_kv_heads = None
    for attr in ["num_key_value_heads", "num_kv_heads"]:
        if hasattr(attn_module, attr):
            num_kv_heads = getattr(attn_module, attr)
            break
    if num_kv_heads is None and hasattr(attn_module, "config"):
        config = attn_module.config
        for attr in ["num_key_value_heads", "num_kv_heads"]:
            if hasattr(config, attr):
                num_kv_heads = getattr(config, attr)
                break
    if num_kv_heads is None:
        num_kv_heads = num_heads

    if num_heads is None or head_dim is None:
        if hasattr(attn_module, "q_proj") and hasattr(attn_module.q_proj, "weight"):
            out_features = attn_module.q_proj.weight.shape[0]
            in_features = attn_module.q_proj.weight.shape[1]

            for candidate_dim in [128, 64, 32, 96, 256]:
                if out_features % candidate_dim == 0:
                    head_dim = head_dim or candidate_dim
                    num_heads = num_heads or (out_features // candidate_dim)
                    break

    if num_heads is None:
        raise ValueError("Could not determine num_heads from attention module")
    if head_dim is None:
        raise ValueError("Could not determine head_dim from attention module")

    return num_heads, num_kv_heads, head_dim


def reset_ecc_cache(model):
    """Reset ECC cache state for a new text/trial.

    Resets both the block manager (cache storage) and backend statistics
    (error counts, injection counts). This ensures fresh state for each
    text within a trial, preventing statistics from leaking across texts.
    """
    if hasattr(model, "_ecc_block_manager"):
        model._ecc_block_manager.reset()
    if hasattr(model, "_ecc_backend"):
        model._ecc_backend.reset_stats()


def get_ecc_stats(model):
    stats = {}
    if hasattr(model, "_ecc_block_manager"):
        manager = model._ecc_block_manager
        stats["allocated_blocks"] = sum(
            len(blocks) for blocks in manager.seq_to_blocks.values()
        )
        stats["free_blocks"] = len(manager.free_blocks)
        stats["sequences"] = len(manager.seq_to_blocks)
    if hasattr(model, "_ecc_backend"):
        backend = model._ecc_backend
        stats["injection_count"] = backend._injection_count
        stats["errors_corrected"] = backend._errors_corrected
        stats["errors_detected"] = backend._errors_detected
        stats["total_values"] = backend._total_values
    return stats
