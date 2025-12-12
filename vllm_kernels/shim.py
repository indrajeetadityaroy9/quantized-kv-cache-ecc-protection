"""
ECC PagedAttention Shim for HuggingFace Models.

Provides a drop-in replacement for HuggingFace attention layers that routes
KV cache operations through ECC-protected Triton kernels.

Components:
- SimpleBlockManager: GPU memory heap for paged KV cache
- ECCBackend: Wrapper for Triton kernel calls
- ECCPagedAttentionShim: Replacement attention module
- patch_model_with_ecc_attention: Context manager for monkey-patching
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from contextlib import contextmanager
from dataclasses import dataclass
import math


@dataclass
class ECCShimConfig:
    """Configuration for ECC PagedAttention shim.

    Supported codecs:
        - "fp16": No quantization, no protection (oracle baseline)
        - "int4": INT4 quantization only, no ECC protection
        - "hamming74": INT4 + Hamming(7,4) SEC (7 bits/value)
        - "hamming84": INT4 + Hamming(8,4) SECDED (8 bits/value)
        - "golay": INT4 triplets + Golay(24,12) (24 bits/3 values)
        - "adaptive": Golay for sink blocks, Hamming84 for context
    """
    codec: str = "hamming84"  # "fp16", "int4", "hamming74", "hamming84", "golay", "adaptive"
    ber: float = 0.0  # Bit error rate for fault injection
    sink_blocks: int = 4  # Number of blocks for Golay in adaptive mode
    block_size: int = 16  # Tokens per cache block
    num_blocks: int = 256  # Total physical blocks
    inject_errors: bool = False  # Whether to inject bit errors
    seed: int = 42  # Random seed for error injection
    use_interpolation: bool = False  # Apply linear interpolation for double-detected errors


class SimpleBlockManager:
    """
    Manages the GPU memory heap for PagedAttention.

    Allocates and tracks physical blocks for sequences. All layers share
    the same physical blocks but write to different layer offsets.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        codec: str = "hamming84",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.codec = codec

        # Compute storage per head based on codec
        if codec == "fp16":
            # FP16: Store raw float16 values (no quantization)
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.float16
            self.needs_scales = False  # FP16 doesn't need quantization scales
        elif codec == "int4":
            # INT4: Store quantized uint8 values (no ECC)
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True
        elif codec == "hamming74":
            # 1:1 mapping for Hamming74 (7 bits stored in uint8)
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True
        elif codec == "hamming84":
            # 1:1 mapping for Hamming84
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True
        elif codec == "golay":
            # 3:1 packing for Golay
            self.values_per_head = block_size * ((head_dim + 2) // 3)
            self.cache_dtype = torch.int32
            self.needs_scales = True
        else:
            # Default: assume ECC with uint8
            self.values_per_head = block_size * head_dim
            self.cache_dtype = torch.uint8
            self.needs_scales = True

        # Backward compatibility alias
        self.codewords_per_head = self.values_per_head

        # Allocate the main KV cache tensors (The Heap)
        # Shape: [num_blocks, num_layers, num_kv_heads, codewords_per_head]
        self.k_cache = torch.zeros(
            num_blocks, num_layers, num_kv_heads, self.codewords_per_head,
            dtype=self.cache_dtype, device=device
        )
        self.v_cache = torch.zeros(
            num_blocks, num_layers, num_kv_heads, self.codewords_per_head,
            dtype=self.cache_dtype, device=device
        )

        # Scales: [num_blocks, num_layers, num_kv_heads, block_size]
        self.k_scales = torch.zeros(
            num_blocks, num_layers, num_kv_heads, block_size,
            dtype=torch.float32, device=device
        )
        self.v_scales = torch.zeros(
            num_blocks, num_layers, num_kv_heads, block_size,
            dtype=torch.float32, device=device
        )

        # For adaptive UEP: separate Golay cache for sink blocks
        if codec == "adaptive":
            golay_codewords = block_size * ((head_dim + 2) // 3)
            self.sink_k_cache = torch.zeros(
                num_blocks, num_layers, num_kv_heads, golay_codewords,
                dtype=torch.int32, device=device
            )
            self.sink_v_cache = torch.zeros(
                num_blocks, num_layers, num_kv_heads, golay_codewords,
                dtype=torch.int32, device=device
            )
            self.sink_k_scales = torch.zeros(
                num_blocks, num_layers, num_kv_heads, block_size,
                dtype=torch.float32, device=device
            )
            self.sink_v_scales = torch.zeros(
                num_blocks, num_layers, num_kv_heads, block_size,
                dtype=torch.float32, device=device
            )
        else:
            self.sink_k_cache = None
            self.sink_v_cache = None

        # Block allocation tracking
        self.free_blocks: List[int] = list(range(num_blocks))
        self.seq_to_blocks: Dict[int, List[int]] = {}  # seq_id -> [block_ids]
        self.seq_to_len: Dict[int, int] = {}  # seq_id -> context_len

        # Block table for kernel interface: [max_seqs, max_blocks_per_seq]
        self.max_seqs = 32
        self.max_blocks_per_seq = num_blocks
        self.block_table = torch.full(
            (self.max_seqs, self.max_blocks_per_seq),
            -1, dtype=torch.int32, device=device
        )

    def allocate(self, seq_id: int, num_tokens: int) -> Tuple[torch.Tensor, int]:
        """
        Allocate blocks for a sequence.

        Args:
            seq_id: Sequence identifier
            num_tokens: Number of tokens to allocate for

        Returns:
            block_table_row: Physical block indices for this sequence
            context_len: Total context length after allocation
        """
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        # Get existing blocks for this sequence
        existing_blocks = self.seq_to_blocks.get(seq_id, [])
        existing_len = self.seq_to_len.get(seq_id, 0)

        # Calculate how many new blocks we need
        existing_block_count = len(existing_blocks)
        new_blocks_needed = max(0, num_blocks_needed - existing_block_count)

        if new_blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Out of blocks: need {new_blocks_needed}, have {len(self.free_blocks)}"
            )

        # Allocate new blocks
        new_blocks = [self.free_blocks.pop(0) for _ in range(new_blocks_needed)]
        all_blocks = existing_blocks + new_blocks

        # Update tracking
        self.seq_to_blocks[seq_id] = all_blocks
        self.seq_to_len[seq_id] = num_tokens

        # Update block table
        for i, block_id in enumerate(all_blocks):
            self.block_table[seq_id, i] = block_id

        return self.block_table[seq_id], num_tokens

    def get_block_table(self, seq_id: int) -> torch.Tensor:
        """Get block table row for a sequence."""
        return self.block_table[seq_id]

    def get_context_len(self, seq_id: int) -> int:
        """Get current context length for a sequence."""
        return self.seq_to_len.get(seq_id, 0)

    def reset(self):
        """Reset all allocations (call between PPL samples)."""
        # Return all blocks to free list
        for blocks in self.seq_to_blocks.values():
            self.free_blocks.extend(blocks)
        self.seq_to_blocks.clear()
        self.seq_to_len.clear()

        # Reset block table
        self.block_table.fill_(-1)

        # Zero out caches (optional, for cleanliness)
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_scales.zero_()
        self.v_scales.zero_()

        if self.sink_k_cache is not None:
            self.sink_k_cache.zero_()
            self.sink_v_cache.zero_()


class ECCBackend:
    """
    Backend wrapper for ECC Triton kernel calls.

    Handles tensor reshaping and kernel invocation for write and attend operations.
    """

    def __init__(
        self,
        manager: SimpleBlockManager,
        config: ECCShimConfig,
        num_heads: int,  # Query heads (may differ from KV heads for GQA)
    ):
        self.manager = manager
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = manager.num_kv_heads
        self.head_dim = manager.head_dim

        # For GQA: how many Q heads per KV head
        self.num_kv_groups = num_heads // self.num_kv_heads

        # Error injection state
        self._injection_count = 0

        # Error statistics counters (aggregated across all decode calls)
        self._errors_corrected = 0  # Single-bit errors corrected (SEC)
        self._errors_detected = 0   # Double-bit errors detected but uncorrectable (DED)

    def write(
        self,
        k: torch.Tensor,  # [batch, seq, num_kv_heads * head_dim] (after RoPE)
        v: torch.Tensor,  # [batch, seq, num_kv_heads * head_dim]
        layer_idx: int,
        seq_id: int = 0,
    ):
        """
        Write K, V to paged cache with optional ECC protection.

        Supports:
        - fp16: No quantization, store raw FP16 values
        - int4: INT4 quantization only, no ECC
        - hamming74: Hamming(7,4) SEC (7-bit codewords)
        - hamming84: Hamming(8,4) SECDED (8-bit codewords)
        - golay: Golay(24,12) with 3:1 packing
        - adaptive: Sink blocks use Golay, context uses Hamming84

        Args:
            k: Key tensor (post-RoPE) [batch, seq, num_kv_heads * head_dim]
            v: Value tensor [batch, seq, num_kv_heads * head_dim]
            layer_idx: Current layer index
            seq_id: Sequence identifier for block allocation
        """
        batch_size, seq_len, hidden_kv = k.shape
        device = k.device
        head_dim = self.head_dim

        # Ensure blocks are allocated
        current_len = self.manager.get_context_len(seq_id)
        if current_len < seq_len:
            self.manager.allocate(seq_id, seq_len)

        block_table = self.manager.get_block_table(seq_id)

        # Reshape to [batch, seq, num_kv_heads, head_dim]
        k_reshaped = k.view(batch_size, seq_len, self.num_kv_heads, head_dim)
        v_reshaped = v.view(batch_size, seq_len, self.num_kv_heads, head_dim)

        # =====================================================================
        # FP16 PATH: No quantization, store raw values
        # =====================================================================
        if self.config.codec == "fp16":
            for b in range(batch_size):
                for pos in range(seq_len):
                    logical_block = pos // self.manager.block_size
                    slot_in_block = pos % self.manager.block_size
                    physical_block = int(block_table[logical_block].item())

                    if physical_block < 0:
                        continue

                    for h in range(self.num_kv_heads):
                        k_vals = k_reshaped[b, pos, h]  # [head_dim] FP16
                        v_vals = v_reshaped[b, pos, h]  # [head_dim] FP16

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[physical_block, layer_idx, h, offset_start:offset_end] = k_vals
                        self.manager.v_cache[physical_block, layer_idx, h, offset_start:offset_end] = v_vals
            return

        # =====================================================================
        # INT4+ PATHS: Quantization required
        # =====================================================================
        from vllm_kernels.paged_cache_ecc import compute_quantization_scales

        # Compute per-position scales
        k_scales = compute_quantization_scales(k_reshaped.float(), dim=-1)  # [B, S, num_kv_heads]
        v_scales = compute_quantization_scales(v_reshaped.float(), dim=-1)  # [B, S, num_kv_heads]

        # Quantize to INT4: round(value / scale) + 8
        k_int4 = (torch.round(k_reshaped.float() / k_scales.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)
        v_int4 = (torch.round(v_reshaped.float() / v_scales.unsqueeze(-1)).clamp(-8, 7) + 8).to(torch.uint8)

        # =====================================================================
        # INT4 PATH: Quantization only, no ECC
        # =====================================================================
        if self.config.codec == "int4":
            for b in range(batch_size):
                for pos in range(seq_len):
                    logical_block = pos // self.manager.block_size
                    slot_in_block = pos % self.manager.block_size
                    physical_block = int(block_table[logical_block].item())

                    if physical_block < 0:
                        continue

                    for h in range(self.num_kv_heads):
                        k_vals = k_int4[b, pos, h]  # [head_dim] uint8
                        v_vals = v_int4[b, pos, h]  # [head_dim] uint8

                        # Inject bit errors into raw INT4 (4 bits per value)
                        if self.config.inject_errors and self.config.ber > 0:
                            from hamming74.triton_kernels import inject_bit_errors_triton
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_vals = inject_bit_errors_triton(k_vals, self.config.ber, 4, seed)
                            v_vals = inject_bit_errors_triton(v_vals, self.config.ber, 4, seed + 1)

                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[physical_block, layer_idx, h, offset_start:offset_end] = k_vals
                        self.manager.v_cache[physical_block, layer_idx, h, offset_start:offset_end] = v_vals

                        # Store scales
                        self.manager.k_scales[physical_block, layer_idx, h, slot_in_block] = k_scales[b, pos, h]
                        self.manager.v_scales[physical_block, layer_idx, h, slot_in_block] = v_scales[b, pos, h]
            return

        # =====================================================================
        # ECC PATHS: Hamming74, Hamming84, Golay, Adaptive
        # =====================================================================
        from hamming74.triton_kernels import (
            hamming74_encode, hamming84_encode, golay_encode, inject_bit_errors_triton
        )

        # Determine if we're using adaptive UEP
        use_adaptive = self.config.codec == "adaptive" and self.config.sink_blocks > 0

        # Precompute Golay padding: head_dim=128 needs to be padded to 129 (multiple of 3)
        golay_padded_dim = ((head_dim + 2) // 3) * 3  # 129 for head_dim=128
        golay_num_codewords = golay_padded_dim // 3  # 43 codewords

        # Encode and store per-block
        for b in range(batch_size):
            for pos in range(seq_len):
                logical_block = pos // self.manager.block_size
                slot_in_block = pos % self.manager.block_size
                physical_block = int(block_table[logical_block].item())

                if physical_block < 0:
                    continue

                # Check if this position is in a sink block (Golay) or context block (Hamming84)
                is_sink_block = use_adaptive and logical_block < self.config.sink_blocks

                for h in range(self.num_kv_heads):
                    # Get INT4 values for this position and head
                    k_vals = k_int4[b, pos, h]  # [head_dim]
                    v_vals = v_int4[b, pos, h]  # [head_dim]

                    if is_sink_block:
                        # =====================================================
                        # GOLAY PATH: Sink blocks get stronger protection
                        # =====================================================
                        # Pad to multiple of 3: 128 → 129
                        k_padded = torch.zeros(golay_padded_dim, dtype=torch.uint8, device=device)
                        v_padded = torch.zeros(golay_padded_dim, dtype=torch.uint8, device=device)
                        k_padded[:head_dim] = k_vals
                        v_padded[:head_dim] = v_vals

                        # Reshape to triplets [num_codewords, 3]
                        k_triplets = k_padded.view(-1, 3)
                        v_triplets = v_padded.view(-1, 3)

                        # Encode with Golay(24,12) - returns int32 codewords
                        k_encoded = golay_encode(k_triplets)  # [num_codewords]
                        v_encoded = golay_encode(v_triplets)  # [num_codewords]

                        # Inject bit errors (24 bits per Golay codeword)
                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(k_encoded, self.config.ber, 24, seed)
                            v_encoded = inject_bit_errors_triton(v_encoded, self.config.ber, 24, seed + 1)

                        # Store in SINK cache
                        offset_start = slot_in_block * golay_num_codewords
                        offset_end = offset_start + golay_num_codewords
                        self.manager.sink_k_cache[physical_block, layer_idx, h, offset_start:offset_end] = k_encoded
                        self.manager.sink_v_cache[physical_block, layer_idx, h, offset_start:offset_end] = v_encoded

                        # Store scales in sink scales
                        self.manager.sink_k_scales[physical_block, layer_idx, h, slot_in_block] = k_scales[b, pos, h]
                        self.manager.sink_v_scales[physical_block, layer_idx, h, slot_in_block] = v_scales[b, pos, h]
                    elif self.config.codec == "hamming74":
                        # =====================================================
                        # HAMMING74 PATH: SEC only (7-bit codewords)
                        # =====================================================
                        k_encoded = hamming74_encode(k_vals)
                        v_encoded = hamming74_encode(v_vals)

                        # Inject bit errors (7 bits per Hamming74 codeword)
                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(k_encoded, self.config.ber, 7, seed)
                            v_encoded = inject_bit_errors_triton(v_encoded, self.config.ber, 7, seed + 1)

                        # Store in main cache
                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[physical_block, layer_idx, h, offset_start:offset_end] = k_encoded
                        self.manager.v_cache[physical_block, layer_idx, h, offset_start:offset_end] = v_encoded

                        # Store scales
                        self.manager.k_scales[physical_block, layer_idx, h, slot_in_block] = k_scales[b, pos, h]
                        self.manager.v_scales[physical_block, layer_idx, h, slot_in_block] = v_scales[b, pos, h]
                    elif self.config.codec == "golay":
                        # =====================================================
                        # GOLAY PATH: Strong protection (corrects up to 3 bits)
                        # =====================================================
                        # Pad to multiple of 3: 128 → 129
                        k_padded = torch.zeros(golay_padded_dim, dtype=torch.uint8, device=device)
                        v_padded = torch.zeros(golay_padded_dim, dtype=torch.uint8, device=device)
                        k_padded[:head_dim] = k_vals
                        v_padded[:head_dim] = v_vals

                        # Reshape to triplets [num_codewords, 3]
                        k_triplets = k_padded.view(-1, 3)
                        v_triplets = v_padded.view(-1, 3)

                        # Encode with Golay(24,12) - returns int32 codewords
                        k_encoded = golay_encode(k_triplets)  # [golay_num_codewords]
                        v_encoded = golay_encode(v_triplets)  # [golay_num_codewords]

                        # Inject bit errors (24 bits per Golay codeword)
                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(k_encoded, self.config.ber, 24, seed)
                            v_encoded = inject_bit_errors_triton(v_encoded, self.config.ber, 24, seed + 1)

                        # Store in main cache (Golay uses int32, different offsets)
                        offset_start = slot_in_block * golay_num_codewords
                        offset_end = offset_start + golay_num_codewords
                        self.manager.k_cache[physical_block, layer_idx, h, offset_start:offset_end] = k_encoded
                        self.manager.v_cache[physical_block, layer_idx, h, offset_start:offset_end] = v_encoded

                        # Store scales
                        self.manager.k_scales[physical_block, layer_idx, h, slot_in_block] = k_scales[b, pos, h]
                        self.manager.v_scales[physical_block, layer_idx, h, slot_in_block] = v_scales[b, pos, h]
                    else:
                        # =====================================================
                        # HAMMING84 PATH: Context blocks use standard protection
                        # =====================================================
                        k_encoded = hamming84_encode(k_vals)
                        v_encoded = hamming84_encode(v_vals)

                        # Inject bit errors (8 bits per Hamming84 codeword)
                        if self.config.inject_errors and self.config.ber > 0:
                            seed = self.config.seed + self._injection_count
                            self._injection_count += 1
                            k_encoded = inject_bit_errors_triton(k_encoded, self.config.ber, 8, seed)
                            v_encoded = inject_bit_errors_triton(v_encoded, self.config.ber, 8, seed + 1)

                        # Store in main cache
                        offset_start = slot_in_block * head_dim
                        offset_end = offset_start + head_dim
                        self.manager.k_cache[physical_block, layer_idx, h, offset_start:offset_end] = k_encoded
                        self.manager.v_cache[physical_block, layer_idx, h, offset_start:offset_end] = v_encoded

                        # Store scales
                        self.manager.k_scales[physical_block, layer_idx, h, slot_in_block] = k_scales[b, pos, h]
                        self.manager.v_scales[physical_block, layer_idx, h, slot_in_block] = v_scales[b, pos, h]

    def attend(
        self,
        q: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        layer_idx: int,
        seq_id: int = 0,
    ) -> torch.Tensor:
        """
        Run attention with optional ECC-protected KV cache.

        Supports:
        - fp16: Direct FP16 values (no decode)
        - int4: INT4 dequantization only (no ECC)
        - hamming74/84: ECC decode + dequantize
        - golay: Golay decode + dequantize
        - adaptive: Golay for sinks, Hamming84 for context

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Current layer index
            seq_id: Sequence identifier

        Returns:
            attn_output: Attention output [batch, num_heads, seq_len, head_dim]
        """
        import torch.nn.functional as F

        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device

        context_len = self.manager.get_context_len(seq_id)
        if context_len == 0:
            return torch.zeros_like(q)

        block_table = self.manager.get_block_table(seq_id)
        num_ctx_blocks = (context_len + self.manager.block_size - 1) // self.manager.block_size

        # ===================================================================
        # FP16 PATH: Direct load, no decode needed
        # ===================================================================
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

                    k_val = self.manager.k_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    v_val = self.manager.v_cache[phys_block, layer_idx, :, offset_start:offset_end]

                    k_list.append(k_val)  # [num_kv_heads, head_dim]
                    v_list.append(v_val)

            if not k_list:
                return torch.zeros_like(q)

            # Stack: [ctx_len, num_kv_heads, head_dim]
            k_float = torch.stack(k_list, dim=0)
            v_float = torch.stack(v_list, dim=0)

            # Handle GQA and run attention
            return self._run_attention(q, k_float, v_float, device)

        # ===================================================================
        # INT4 PATH: Dequantize only, no ECC decode
        # ===================================================================
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

                    k_val = self.manager.k_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    v_val = self.manager.v_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                    k_list.append(k_val)
                    v_list.append(v_val)
                    k_scale_list.append(k_scale)
                    v_scale_list.append(v_scale)

            if not k_list:
                return torch.zeros_like(q)

            # Stack: [ctx_len, num_kv_heads, head_dim]
            k_int4 = torch.stack(k_list, dim=0)
            v_int4 = torch.stack(v_list, dim=0)
            k_scales = torch.stack(k_scale_list, dim=0)  # [ctx_len, num_kv_heads]
            v_scales = torch.stack(v_scale_list, dim=0)

            # Dequantize: (int4 - 8) * scale
            k_float = (k_int4.float() - 8.0) * k_scales.unsqueeze(-1)
            v_float = (v_int4.float() - 8.0) * v_scales.unsqueeze(-1)

            return self._run_attention(q, k_float, v_float, device)

        # ===================================================================
        # ECC PATHS: Hamming74, Hamming84, Golay, Adaptive
        # ===================================================================
        from hamming74.triton_kernels import (
            hamming74_decode, hamming84_decode, golay_decode, interpolate_double_errors
        )
        from hamming74.triton_kernels.config import ErrorType

        # Determine if we're using adaptive UEP
        use_adaptive = self.config.codec == "adaptive" and self.config.sink_blocks > 0

        # Calculate sink boundary in tokens
        sink_tokens = self.config.sink_blocks * self.manager.block_size if use_adaptive else 0
        sink_tokens = min(sink_tokens, context_len)  # Can't have more sink tokens than context

        # Golay layout: padded head_dim to multiple of 3
        golay_padded_dim = ((head_dim + 2) // 3) * 3  # 129 for head_dim=128
        golay_num_codewords = golay_padded_dim // 3   # 43 codewords

        # Lists for sink tokens (Golay-encoded)
        sink_k_enc_list = []
        sink_v_enc_list = []
        sink_k_scale_list = []
        sink_v_scale_list = []

        # Lists for context tokens (Hamming-encoded)
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

            # Determine if this is a sink block or context block
            is_sink_block = use_adaptive and blk_idx < self.config.sink_blocks
            # Standalone Golay mode: all blocks use Golay encoding in main cache
            is_standalone_golay = self.config.codec == "golay"

            for slot in range(end_pos - start_pos):
                global_pos = start_pos + slot

                if is_sink_block:
                    # Collect from SINK cache (Golay-encoded, adaptive mode)
                    offset_start = slot * golay_num_codewords
                    offset_end = offset_start + golay_num_codewords

                    k_enc = self.manager.sink_k_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    v_enc = self.manager.sink_v_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    k_scale = self.manager.sink_k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.sink_v_scales[phys_block, layer_idx, :, slot]

                    sink_k_enc_list.append(k_enc)  # [num_kv_heads, golay_num_codewords]
                    sink_v_enc_list.append(v_enc)
                    sink_k_scale_list.append(k_scale)  # [num_kv_heads]
                    sink_v_scale_list.append(v_scale)
                elif is_standalone_golay:
                    # Collect from MAIN cache (Golay-encoded, standalone mode)
                    # Uses same offsets as sink but stored in main cache
                    offset_start = slot * golay_num_codewords
                    offset_end = offset_start + golay_num_codewords

                    k_enc = self.manager.k_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    v_enc = self.manager.v_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                    # Add to sink lists for Golay decode (reuse same decode path)
                    sink_k_enc_list.append(k_enc)  # [num_kv_heads, golay_num_codewords]
                    sink_v_enc_list.append(v_enc)
                    sink_k_scale_list.append(k_scale)  # [num_kv_heads]
                    sink_v_scale_list.append(v_scale)
                else:
                    # Collect from MAIN cache (Hamming-encoded)
                    offset_start = slot * head_dim
                    offset_end = offset_start + head_dim

                    k_enc = self.manager.k_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    v_enc = self.manager.v_cache[phys_block, layer_idx, :, offset_start:offset_end]
                    k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                    v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                    ctx_k_enc_list.append(k_enc)  # [num_kv_heads, head_dim]
                    ctx_v_enc_list.append(v_enc)
                    ctx_k_scale_list.append(k_scale)  # [num_kv_heads]
                    ctx_v_scale_list.append(v_scale)

        # ===================================================================
        # STEP 2: Bulk decode SINK tokens with Golay (one call!)
        # ===================================================================
        sink_k_float = None
        sink_v_float = None

        if sink_k_enc_list:
            # Stack: [sink_len, num_kv_heads, golay_num_codewords]
            sink_k_stacked = torch.stack(sink_k_enc_list, dim=0)
            sink_v_stacked = torch.stack(sink_v_enc_list, dim=0)
            sink_k_scale_stacked = torch.stack(sink_k_scale_list, dim=0)  # [sink_len, num_kv_heads]
            sink_v_scale_stacked = torch.stack(sink_v_scale_list, dim=0)

            sink_len = sink_k_stacked.shape[0]

            # Flatten, decode, reshape - ONE decode call for all sink data
            # golay_decode expects [N] int32 codewords, returns [N, 3] triplets
            sink_k_dec_triplets, k_golay_stats = golay_decode(sink_k_stacked.flatten())
            sink_v_dec_triplets, v_golay_stats = golay_decode(sink_v_stacked.flatten())

            # Aggregate Golay statistics (corrected, uncorrectable)
            self._errors_corrected += k_golay_stats[0] + v_golay_stats[0]
            self._errors_detected += k_golay_stats[1] + v_golay_stats[1]  # uncorrectable as detected

            # Reshape: flatten triplets to head_dim, then slice to remove padding
            # Each position has: num_kv_heads * golay_num_codewords codewords
            # Each codeword decodes to 3 values, so total = num_kv_heads * golay_num_codewords * 3
            # = num_kv_heads * golay_padded_dim values per position
            total_decoded_per_pos = self.num_kv_heads * golay_padded_dim

            # Flatten triplets: [N, 3] -> [N*3]
            sink_k_dec_flat = sink_k_dec_triplets.flatten()
            sink_v_dec_flat = sink_v_dec_triplets.flatten()

            # Reshape to [sink_len, num_kv_heads, golay_padded_dim]
            sink_k_dec = sink_k_dec_flat.view(sink_len, self.num_kv_heads, golay_padded_dim)
            sink_v_dec = sink_v_dec_flat.view(sink_len, self.num_kv_heads, golay_padded_dim)

            # CRITICAL: Slice to remove padding: 129 -> 128
            sink_k_dec = sink_k_dec[:, :, :head_dim]
            sink_v_dec = sink_v_dec[:, :, :head_dim]

            # Dequantize
            sink_k_float = (sink_k_dec.float() - 8.0) * sink_k_scale_stacked.unsqueeze(-1)
            sink_v_float = (sink_v_dec.float() - 8.0) * sink_v_scale_stacked.unsqueeze(-1)

        # ===================================================================
        # STEP 3: Bulk decode CONTEXT tokens (Hamming74 or Hamming84)
        # ===================================================================
        ctx_k_float = None
        ctx_v_float = None

        if ctx_k_enc_list:
            # Stack: [ctx_len, num_kv_heads, head_dim]
            ctx_k_stacked = torch.stack(ctx_k_enc_list, dim=0)
            ctx_v_stacked = torch.stack(ctx_v_enc_list, dim=0)
            ctx_k_scale_stacked = torch.stack(ctx_k_scale_list, dim=0)  # [ctx_len, num_kv_heads]
            ctx_v_scale_stacked = torch.stack(ctx_v_scale_list, dim=0)

            ctx_len_actual = ctx_k_stacked.shape[0]

            # Flatten for bulk decode
            k_flat = ctx_k_stacked.flatten()
            v_flat = ctx_v_stacked.flatten()

            if self.config.codec == "hamming74":
                # Hamming(7,4) SEC decode - returns (errors_corrected_count,)
                ctx_k_dec, k_h74_stats = hamming74_decode(k_flat)
                ctx_v_dec, v_h74_stats = hamming74_decode(v_flat)

                # Aggregate Hamming74 statistics (corrected only, no DED)
                self._errors_corrected += k_h74_stats[0] + v_h74_stats[0]
            elif self.config.use_interpolation or self.config.codec == "adaptive":
                # Hamming(8,4) SECDED decode with interpolation for double errors
                # Adaptive mode uses interpolation for context blocks to avoid zero-out degradation
                ctx_k_dec, k_error_types, k_h84_stats = hamming84_decode(k_flat, return_error_types=True)
                ctx_v_dec, v_error_types, v_h84_stats = hamming84_decode(v_flat, return_error_types=True)

                # Aggregate Hamming84 statistics (corrected, detected)
                self._errors_corrected += k_h84_stats[0] + v_h84_stats[0]
                self._errors_detected += k_h84_stats[1] + v_h84_stats[1]

                # Apply linear interpolation for double-detected errors
                ctx_k_dec = interpolate_double_errors(ctx_k_dec, k_error_types)
                ctx_v_dec = interpolate_double_errors(ctx_v_dec, v_error_types)
            else:
                # Standard Hamming(8,4) SECDED decode
                ctx_k_dec, k_h84_stats = hamming84_decode(k_flat)
                ctx_v_dec, v_h84_stats = hamming84_decode(v_flat)

                # Aggregate Hamming84 statistics (corrected, detected)
                self._errors_corrected += k_h84_stats[0] + v_h84_stats[0]
                self._errors_detected += k_h84_stats[1] + v_h84_stats[1]

            ctx_k_dec = ctx_k_dec.view(ctx_len_actual, self.num_kv_heads, head_dim)
            ctx_v_dec = ctx_v_dec.view(ctx_len_actual, self.num_kv_heads, head_dim)

            # Dequantize
            ctx_k_float = (ctx_k_dec.float() - 8.0) * ctx_k_scale_stacked.unsqueeze(-1)
            ctx_v_float = (ctx_v_dec.float() - 8.0) * ctx_v_scale_stacked.unsqueeze(-1)

        # ===================================================================
        # STEP 4: Concatenate sink + context KV
        # ===================================================================
        if sink_k_float is not None and ctx_k_float is not None:
            # Ensure same dtype and device before concatenating
            k_float = torch.cat([sink_k_float.to(device), ctx_k_float.to(device)], dim=0)
            v_float = torch.cat([sink_v_float.to(device), ctx_v_float.to(device)], dim=0)
        elif sink_k_float is not None:
            k_float = sink_k_float.to(device)
            v_float = sink_v_float.to(device)
        elif ctx_k_float is not None:
            k_float = ctx_k_float.to(device)
            v_float = ctx_v_float.to(device)
        else:
            return torch.zeros_like(q)

        # ===================================================================
        # STEP 5: Handle GQA - expand KV heads to match Q heads
        # ===================================================================
        if self.num_kv_groups > 1:
            # GQA: [ctx, num_kv_heads, dim] -> [ctx, num_heads, dim]
            k_float = k_float.repeat_interleave(self.num_kv_groups, dim=1)
            v_float = v_float.repeat_interleave(self.num_kv_groups, dim=1)

        # ===================================================================
        # STEP 6: Reshape for SDPA and run optimized attention
        # ===================================================================
        # Reshape: [ctx, heads, dim] -> [batch, heads, ctx, dim]
        k_for_sdpa = k_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
        v_for_sdpa = v_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

        # Use F.scaled_dot_product_attention with is_causal=True
        # This handles causal masking efficiently on GPU!
        # Note: For prefill, seq_len == ctx_len (we wrote then query same sequence)
        output = F.scaled_dot_product_attention(
            q, k_for_sdpa, v_for_sdpa,
            is_causal=True,  # CRITICAL: efficient causal masking
        )

        return output  # [batch, num_heads, seq_len, head_dim]

    def _run_attention(
        self,
        q: torch.Tensor,
        k_float: torch.Tensor,
        v_float: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Run scaled dot-product attention with GQA handling.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k_float: Key tensor [ctx_len, num_kv_heads, head_dim]
            v_float: Value tensor [ctx_len, num_kv_heads, head_dim]
            device: Target device

        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
        """
        import torch.nn.functional as F

        # Handle GQA - expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k_float = k_float.repeat_interleave(self.num_kv_groups, dim=1)
            v_float = v_float.repeat_interleave(self.num_kv_groups, dim=1)

        # Reshape: [ctx, heads, dim] -> [batch, heads, ctx, dim]
        k_for_sdpa = k_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
        v_for_sdpa = v_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

        # Use F.scaled_dot_product_attention with is_causal=True
        output = F.scaled_dot_product_attention(
            q, k_for_sdpa, v_for_sdpa,
            is_causal=True,
        )

        return output


class ECCPagedAttentionShim(nn.Module):
    """
    Shim that replaces HuggingFace attention with ECC-protected PagedAttention.

    Designed for LLaMA models. Handles:
    - RoPE application before cache write
    - GQA (different num_heads for Q vs KV)
    - Proper return format for HuggingFace compatibility
    """

    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        backend: ECCBackend,
        rotary_emb: nn.Module,
    ):
        super().__init__()

        # Keep original projection layers
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Store attention parameters using robust extraction
        # Import helper function - it's defined later in this file but available at runtime
        num_heads, num_kv_heads, head_dim = _get_attention_params(original_attn)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = self.num_heads * self.head_dim

        # Backend and layer info
        self.backend = backend
        self.layer_idx = layer_idx
        self.rotary_emb = rotary_emb

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with ECC-protected KV cache.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Attention mask (optional)
            position_ids: Position indices for RoPE
            past_key_value: Ignored (we use internal paged cache)
            output_attentions: Whether to return attention weights
            use_cache: Whether to return updated cache

        Returns:
            output: Attention output [batch, seq_len, hidden_size]
            past_key_value: Dummy tuple (empty tensors) for HF compatibility
            attn_weights: None (not supported)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)  # [B, S, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)  # [B, S, num_kv_heads * head_dim]

        # 2. Reshape for RoPE
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Now: Q [B, num_heads, S, D], K/V [B, num_kv_heads, S, D]

        # 3. Handle position_ids (CRITICAL: may be None!)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 4. Apply RoPE
        # LlamaRotaryEmbedding returns cos/sin with shape [1, 1, seq_len, head_dim]
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # 5. Write RoPE'd K, V to ECC cache
        # Reshape back to [B, S, num_kv_heads * head_dim] for write
        k_flat = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        v_flat = v.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        self.backend.write(k_flat, v_flat, self.layer_idx, seq_id=0)

        # 6. Run ECC attention
        # Q shape: [B, num_heads, S, D]
        attn_output = self.backend.attend(q, self.layer_idx, seq_id=0)

        # 7. Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, num_heads, D]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        # 8. Return with format matching HuggingFace LlamaAttention
        # Newer transformers versions expect (output, attn_weights) when output_attentions=False
        # and (output, attn_weights, past_key_value) when use_cache=True
        # The decoder layer does: hidden_states, _ = self.self_attn(...) expecting 2 values
        # But if use_cache=True, it expects 3 values

        if not output_attentions and not use_cache:
            # Standard case: decoder expects (output, attn_weights)
            return output, None
        elif use_cache:
            # Cache case: decoder expects (output, attn_weights, past_kv)
            dummy_past = None  # We manage our own cache
            return output, None, dummy_past
        else:
            # output_attentions case
            return output, None

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embedding to Q and K.

        Handles different cos/sin shapes from various transformers versions.
        Q shape: [batch, num_heads, seq_len, head_dim]
        K shape: [batch, num_kv_heads, seq_len, head_dim]  (may differ for GQA)
        """
        # Rotate half implementation
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Get shapes - Q and K may have different number of heads (GQA)
        batch, q_heads, seq_len, head_dim = q.shape
        _, k_heads, _, _ = k.shape

        # Reshape cos/sin to have 4 dimensions
        # Different versions produce: [seq, dim], [1, seq, dim], [1, 1, seq, dim], [batch, seq, dim]
        while cos.dim() < 4:
            cos = cos.unsqueeze(0 if cos.dim() < 2 else 1)
            sin = sin.unsqueeze(0 if sin.dim() < 2 else 1)

        # cos/sin now have shape [?, ?, seq, dim]
        # For Q: expand to [batch, q_heads, seq, dim]
        # For K: expand to [batch, k_heads, seq, dim]

        # RoPE is applied per-head, so we need cos/sin to broadcast over heads
        # The head dimension (dim 1) should either be 1 (broadcast) or match exactly
        if cos.shape[1] != 1 and cos.shape[1] != q_heads:
            # Unexpected shape, try to squeeze extra dims
            cos = cos[:, :1, :, :]
            sin = sin[:, :1, :, :]

        # Apply to Q
        q_embed = (q * cos) + (rotate_half(q) * sin)

        # Apply to K (handles case where k_heads != q_heads for GQA)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        return q_embed, k_embed


@contextmanager
def patch_model_with_ecc_attention(
    model: nn.Module,
    config: ECCShimConfig,
    num_blocks: int = 256,
):
    """
    Context manager to replace HuggingFace attention with ECC-protected attention.

    Usage:
        with patch_model_with_ecc_attention(model, config):
            outputs = model(input_ids)

    Args:
        model: HuggingFace model (LLaMA)
        config: ECC shim configuration
        num_blocks: Number of physical cache blocks

    Yields:
        Patched model with ECC attention
    """
    # Detect model architecture
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LlamaForCausalLM structure
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, 'layers'):
        # LlamaModel structure
        layers = model.layers
        embed_tokens = model.embed_tokens
    else:
        raise ValueError("Unsupported model architecture")

    # Extract model config using robust attribute extraction
    num_layers = len(layers)
    first_attn = layers[0].self_attn
    num_heads, num_kv_heads, head_dim = _get_attention_params(first_attn)

    # Create block manager (shared across all layers)
    manager = SimpleBlockManager(
        num_blocks=num_blocks,
        block_size=config.block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=next(model.parameters()).device,
        codec=config.codec,
    )

    # Create backend
    backend = ECCBackend(manager, config, num_heads)

    # Store original attention layers
    original_attns = {}  # Use dict to handle partial replacement

    # Find rotary embedding - may be in different locations depending on transformers version
    rotary_emb = _find_rotary_embedding(model, layers)

    try:
        # Replace attention layers with shims
        for layer_idx, layer in enumerate(layers):
            original_attns[layer_idx] = layer.self_attn

            # Create shim
            shim = ECCPagedAttentionShim(
                original_attn=layer.self_attn,
                layer_idx=layer_idx,
                backend=backend,
                rotary_emb=rotary_emb,
            )

            # Replace
            layer.self_attn = shim

        # Attach manager to model for access
        model._ecc_block_manager = manager
        model._ecc_backend = backend

        yield model

    finally:
        # Restore original attention layers
        for layer_idx, original_attn in original_attns.items():
            layers[layer_idx].self_attn = original_attn

        # Clean up
        if hasattr(model, '_ecc_block_manager'):
            delattr(model, '_ecc_block_manager')
        if hasattr(model, '_ecc_backend'):
            delattr(model, '_ecc_backend')


# =============================================================================
# Utility Functions
# =============================================================================

def _find_rotary_embedding(model: nn.Module, layers: nn.ModuleList) -> nn.Module:
    """
    Find the rotary embedding module in a HuggingFace model.

    Searches multiple locations as different transformers versions store it differently:
    - layer.self_attn.rotary_emb (older transformers)
    - layer.rotary_emb
    - model.model.rotary_emb
    - Creates a new one if not found

    Returns:
        The rotary embedding module
    """
    # Try layer.self_attn.rotary_emb (older transformers)
    if hasattr(layers[0].self_attn, 'rotary_emb'):
        return layers[0].self_attn.rotary_emb

    # Try layer.rotary_emb
    if hasattr(layers[0], 'rotary_emb'):
        return layers[0].rotary_emb

    # Try model.model.rotary_emb (some LLaMA versions)
    if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
        return model.model.rotary_emb

    # Try model.rotary_emb
    if hasattr(model, 'rotary_emb'):
        return model.rotary_emb

    # Create a new rotary embedding if not found
    # This handles newer transformers versions where rotary is computed on-the-fly
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        # Get config from model
        config = getattr(model, 'config', None)
        if config is None and hasattr(model, 'model'):
            config = getattr(model.model, 'config', None)

        if config is not None:
            device = next(model.parameters()).device

            # Try newer API (config-based)
            try:
                rotary_emb = LlamaRotaryEmbedding(config=config, device=device)
                return rotary_emb
            except TypeError:
                # Older API (dim-based)
                head_dim = getattr(config, 'head_dim', None)
                if head_dim is None:
                    hidden_size = config.hidden_size
                    num_heads = config.num_attention_heads
                    head_dim = hidden_size // num_heads

                max_position_embeddings = getattr(config, 'max_position_embeddings', 4096)
                rope_theta = getattr(config, 'rope_theta', 10000.0)

                return LlamaRotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                ).to(device)
    except (ImportError, Exception) as e:
        pass

    # Final fallback: create a simple rotary embedding
    class SimpleRotaryEmbedding(nn.Module):
        """Simple RoPE implementation for fallback."""
        def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096):
            super().__init__()
            self.dim = dim
            self.base = base
            self.max_seq_len = max_seq_len

            # Precompute frequencies
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, x, position_ids):
            # x is just for device/dtype, position_ids has shape [batch, seq_len]
            seq_len = position_ids.shape[-1]
            device = position_ids.device

            # Ensure inv_freq is on correct device
            inv_freq = self.inv_freq.to(device)

            # Compute frequencies for each position
            freqs = torch.einsum('i,j->ij', position_ids[0].float(), inv_freq)

            # Create cos and sin
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
            sin = emb.sin().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]

            return cos, sin

    # Get head_dim from first attention layer
    num_heads, num_kv_heads, head_dim = _get_attention_params(layers[0].self_attn)
    device = next(model.parameters()).device

    return SimpleRotaryEmbedding(dim=head_dim).to(device)


def _get_attention_params(attn_module: nn.Module) -> Tuple[int, int, int]:
    """
    Extract attention parameters from a HuggingFace attention module.

    Handles different transformers versions which may use different attribute names.

    Returns:
        (num_heads, num_kv_heads, head_dim)
    """
    # Try to get num_heads
    num_heads = None
    for attr in ['num_heads', 'num_attention_heads']:
        if hasattr(attn_module, attr):
            num_heads = getattr(attn_module, attr)
            break
    if num_heads is None and hasattr(attn_module, 'config'):
        config = attn_module.config
        for attr in ['num_attention_heads', 'num_heads', 'n_head']:
            if hasattr(config, attr):
                num_heads = getattr(config, attr)
                break

    # Try to get head_dim
    head_dim = None
    for attr in ['head_dim', 'head_size']:
        if hasattr(attn_module, attr):
            head_dim = getattr(attn_module, attr)
            break
    if head_dim is None and hasattr(attn_module, 'config'):
        config = attn_module.config
        if hasattr(config, 'head_dim'):
            head_dim = config.head_dim
        elif hasattr(config, 'hidden_size') and num_heads:
            head_dim = config.hidden_size // num_heads

    # Try to get num_kv_heads (for GQA)
    num_kv_heads = None
    for attr in ['num_key_value_heads', 'num_kv_heads']:
        if hasattr(attn_module, attr):
            num_kv_heads = getattr(attn_module, attr)
            break
    if num_kv_heads is None and hasattr(attn_module, 'config'):
        config = attn_module.config
        for attr in ['num_key_value_heads', 'num_kv_heads']:
            if hasattr(config, attr):
                num_kv_heads = getattr(config, attr)
                break
    if num_kv_heads is None:
        num_kv_heads = num_heads  # Default: no GQA

    # Fallback: infer from weight shapes if still None
    if num_heads is None or head_dim is None:
        if hasattr(attn_module, 'q_proj') and hasattr(attn_module.q_proj, 'weight'):
            out_features = attn_module.q_proj.weight.shape[0]
            in_features = attn_module.q_proj.weight.shape[1]
            # Assume head_dim is a common value like 64, 128
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


def reset_ecc_cache(model: nn.Module):
    """
    Reset the ECC cache between PPL samples.

    Call this after processing each document to avoid OOM.
    """
    if hasattr(model, '_ecc_block_manager'):
        model._ecc_block_manager.reset()


def get_ecc_stats(model: nn.Module) -> Dict[str, Any]:
    """Get statistics from the ECC backend.

    Returns:
        Dictionary containing:
        - allocated_blocks: Number of blocks currently allocated
        - free_blocks: Number of blocks available
        - sequences: Number of active sequences
        - injection_count: Number of error injection calls
        - errors_corrected: Total single-bit errors corrected (SEC)
        - errors_detected: Total multi-bit errors detected but uncorrectable (DED)
    """
    stats = {}
    if hasattr(model, '_ecc_block_manager'):
        manager = model._ecc_block_manager
        stats['allocated_blocks'] = sum(len(blocks) for blocks in manager.seq_to_blocks.values())
        stats['free_blocks'] = len(manager.free_blocks)
        stats['sequences'] = len(manager.seq_to_blocks)
    if hasattr(model, '_ecc_backend'):
        backend = model._ecc_backend
        stats['injection_count'] = backend._injection_count
        stats['errors_corrected'] = backend._errors_corrected
        stats['errors_detected'] = backend._errors_detected
    return stats
