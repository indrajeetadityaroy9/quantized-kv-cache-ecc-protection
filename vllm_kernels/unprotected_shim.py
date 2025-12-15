"""
Unprotected KV cache shim for fair comparison with ECC-protected caches.

This module provides UnprotectedBackend and UnprotectedPagedAttentionShim that
mirror the ECC shim architecture but without error correction. This enables
fair comparison: same INT4 quantization, same block-based storage, only
difference is ECC protection vs no protection.

Data flow:
    FP16 K,V → Quantize to INT4 → Store in cache → Read from cache
    → INJECT BIT ERRORS → Dequantize → Attention

Error injection happens AFTER cache read, BEFORE dequantize to simulate
realistic memory corruption in stored data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
import math

from vllm_kernels.paged_cache_ecc import compute_quantization_scales
from vllm_kernels.shim import (
    SimpleBlockManager,
    _get_attention_params,
    _find_rotary_embedding,
)
from hamming74.triton_kernels import inject_bit_errors_triton


class UnprotectedDummyCache:
    """Dummy cache that satisfies the transformers cache interface.

    The unprotected shim manages its own KV cache internally, so this is just
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


class UnprotectedShimConfig:
    """Configuration for unprotected KV cache."""

    def __init__(
        self,
        ber=0.0,
        block_size=16,
        num_blocks=256,
        inject_errors=False,
        seed=42,
    ):
        """
        Args:
            ber: Bit error rate for testing (0.0 = no errors)
            block_size: Tokens per block
            num_blocks: Total cache blocks
            inject_errors: Enable error injection
            seed: Random seed for reproducibility
        """
        self.ber = ber
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.inject_errors = inject_errors
        self.seed = seed
        # Use 'int4' codec for SimpleBlockManager (no ECC)
        self.codec = "int4"


class UnprotectedBackend:
    """
    KV cache backend WITHOUT ECC protection.

    Same as ECCBackend but:
    - No ECC encoding on write
    - No ECC decoding on read
    - Error injection happens on raw INT4 data (simulating memory errors)

    This enables fair comparison with ECC-protected caches since both use
    the same INT4 quantization scheme.
    """

    def __init__(self, manager, config, num_heads):
        self.manager = manager
        self.config = config
        self.num_heads = num_heads
        self.num_kv_heads = manager.num_kv_heads
        self.head_dim = manager.head_dim
        self.num_kv_groups = num_heads // self.num_kv_heads
        self._injection_count = 0
        self._bits_flipped = 0
        self._total_bits = 0

    def write(self, k, v, layer_idx, seq_id=0):
        """
        Store K,V in cache with INT4 quantization (no ECC).

        Flow: FP16 → quantize → INT4 → store in cache

        Args:
            k: Key tensor [batch, seq_len, hidden_kv]
            v: Value tensor [batch, seq_len, hidden_kv]
            layer_idx: Layer index
            seq_id: Sequence ID (default 0)
        """
        batch_size, seq_len, hidden_kv = k.shape
        device = k.device
        head_dim = self.head_dim

        # Allocate blocks if needed
        current_len = self.manager.get_context_len(seq_id)
        if current_len < seq_len:
            self.manager.allocate(seq_id, seq_len)

        block_table = self.manager.get_block_table(seq_id)

        # Reshape to [batch, seq, num_kv_heads, head_dim]
        k_reshaped = k.view(batch_size, seq_len, self.num_kv_heads, head_dim)
        v_reshaped = v.view(batch_size, seq_len, self.num_kv_heads, head_dim)

        # Compute quantization scales (per-token, per-head)
        k_scales = compute_quantization_scales(k_reshaped.float(), dim=-1)
        v_scales = compute_quantization_scales(v_reshaped.float(), dim=-1)

        # Quantize to INT4: value = round(v/scale).clamp(-8, 7) + 8
        # This maps to unsigned 0-15 range for storage
        k_int4 = (
            torch.round(k_reshaped.float() / k_scales.unsqueeze(-1)).clamp(-8, 7) + 8
        ).to(torch.uint8)
        v_int4 = (
            torch.round(v_reshaped.float() / v_scales.unsqueeze(-1)).clamp(-8, 7) + 8
        ).to(torch.uint8)

        # Store in cache (no ECC encoding)
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

                    # Store INT4 values directly (no ECC encoding)
                    offset_start = slot_in_block * head_dim
                    offset_end = offset_start + head_dim
                    self.manager.k_cache[
                        physical_block, layer_idx, h, offset_start:offset_end
                    ] = k_vals
                    self.manager.v_cache[
                        physical_block, layer_idx, h, offset_start:offset_end
                    ] = v_vals

                    # Store scales for dequantization
                    self.manager.k_scales[
                        physical_block, layer_idx, h, slot_in_block
                    ] = k_scales[b, pos, h]
                    self.manager.v_scales[
                        physical_block, layer_idx, h, slot_in_block
                    ] = v_scales[b, pos, h]

    def attend(self, q, layer_idx, seq_id=0):
        """
        Compute attention using cached K,V with optional error injection.

        Flow: read from cache → INJECT BIT ERRORS → dequantize → attention

        Error injection happens AFTER cache read, BEFORE dequantize.
        This simulates realistic memory corruption in stored data.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            seq_id: Sequence ID (default 0)

        Returns:
            Attention output [batch, num_heads, seq_len, head_dim]
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

        # Read INT4 values and scales from cache
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
                ].clone()
                v_val = self.manager.v_cache[
                    phys_block, layer_idx, :, offset_start:offset_end
                ].clone()
                k_scale = self.manager.k_scales[phys_block, layer_idx, :, slot]
                v_scale = self.manager.v_scales[phys_block, layer_idx, :, slot]

                k_list.append(k_val)
                v_list.append(v_val)
                k_scale_list.append(k_scale)
                v_scale_list.append(v_scale)

        if not k_list:
            return torch.zeros_like(q)

        k_int4 = torch.stack(k_list, dim=0)  # [context_len, num_kv_heads, head_dim]
        v_int4 = torch.stack(v_list, dim=0)
        k_scales = torch.stack(k_scale_list, dim=0)  # [context_len, num_kv_heads]
        v_scales = torch.stack(v_scale_list, dim=0)

        # INJECT BIT ERRORS on raw INT4 data (before dequantization)
        # This is where memory corruption would occur in stored data
        if self.config.inject_errors and self.config.ber > 0:
            seed = self.config.seed + self._injection_count
            self._injection_count += 1

            # Track total bits for statistics
            total_bits = k_int4.numel() * 4  # 4 bits per INT4 value
            self._total_bits += total_bits * 2  # k and v

            # Inject errors into INT4 values (4 bits per value)
            k_flat = k_int4.flatten()
            v_flat = v_int4.flatten()

            k_corrupted, k_stats = inject_bit_errors_triton(
                k_flat, self.config.ber, n_bits=4, seed=seed, return_stats=True
            )
            v_corrupted, v_stats = inject_bit_errors_triton(
                v_flat, self.config.ber, n_bits=4, seed=seed + 1, return_stats=True
            )

            k_bits_flipped = k_stats[0] if k_stats else 0
            v_bits_flipped = v_stats[0] if v_stats else 0
            self._bits_flipped += k_bits_flipped + v_bits_flipped

            k_int4 = k_corrupted.view(k_int4.shape)
            v_int4 = v_corrupted.view(v_int4.shape)

        # Dequantize: (int4 - 8) * scale
        k_float = (k_int4.float() - 8.0) * k_scales.unsqueeze(-1)
        v_float = (v_int4.float() - 8.0) * v_scales.unsqueeze(-1)

        return self._run_attention(q, k_float, v_float, device)

    def _run_attention(self, q, k_float, v_float, device):
        """Run scaled dot-product attention."""
        # Expand for GQA if num_kv_heads < num_heads
        if self.num_kv_groups > 1:
            k_float = k_float.repeat_interleave(self.num_kv_groups, dim=1)
            v_float = v_float.repeat_interleave(self.num_kv_groups, dim=1)

        # Reshape for SDPA: [batch, heads, seq, head_dim]
        k_for_sdpa = k_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
        v_for_sdpa = v_float.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

        output = F.scaled_dot_product_attention(
            q,
            k_for_sdpa,
            v_for_sdpa,
            is_causal=True,
        )

        return output


class UnprotectedPagedAttentionShim(nn.Module):
    """
    Attention wrapper using unprotected (no ECC) KV cache.

    Mirrors ECCPagedAttentionShim exactly, but uses UnprotectedBackend.
    This enables fair comparison with ECC-protected attention.
    """

    def __init__(self, original_attn, layer_idx, backend, rotary_emb):
        super().__init__()

        # Validate and copy projection layers
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

        # Get attention parameters
        num_heads, num_kv_heads, head_dim = _get_attention_params(original_attn)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = self.num_heads * self.head_dim

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
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply rotary position embeddings
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            )

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Flatten for cache write
        k_flat = k.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        v_flat = v.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 5. Write to cache (quantize + store, no ECC)
        self.backend.write(k_flat, v_flat, self.layer_idx, seq_id=0)

        # 6. Read from cache with error injection + attention
        attn_output = self.backend.attend(q, self.layer_idx, seq_id=0)

        # 7. Project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        # Return format matching LlamaAttention
        attn_weights = None

        if use_cache:
            num_layers = self.backend.manager.num_layers
            past_key_value = UnprotectedDummyCache(num_layers=num_layers)
        else:
            past_key_value = None

        return output, attn_weights, past_key_value

    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings to Q and K."""
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
def patch_model_with_unprotected_attention(model, config, num_blocks=256):
    """
    Patch model with unprotected (no ECC) paged attention.

    Mirrors patch_model_with_ecc_attention() exactly, but uses UnprotectedBackend.

    Args:
        model: HuggingFace model to patch
        config: UnprotectedShimConfig instance
        num_blocks: Number of cache blocks to allocate

    Yields:
        Patched model with unprotected KV cache

    Example:
        config = UnprotectedShimConfig(ber=1e-4, inject_errors=True)
        with patch_model_with_unprotected_attention(model, config) as patched_model:
            outputs = patched_model(**inputs)
    """
    # Find model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, "layers"):
        layers = model.layers
        embed_tokens = model.embed_tokens
    else:
        raise ValueError("Unsupported model architecture")

    num_layers = len(layers)
    first_attn = layers[0].self_attn

    # Debug output
    print(f"[Unprotected Shim] Patching {num_layers} layers")
    print(f"[Unprotected Shim] Attention type: {type(first_attn).__name__}")
    print(f"[Unprotected Shim] Has q_proj: {hasattr(first_attn, 'q_proj')}")
    print(f"[Unprotected Shim] Has k_proj: {hasattr(first_attn, 'k_proj')}")
    print(f"[Unprotected Shim] Has v_proj: {hasattr(first_attn, 'v_proj')}")

    # Get attention parameters
    num_heads, num_kv_heads, head_dim = _get_attention_params(first_attn)
    print(
        f"[Unprotected Shim] num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
    )

    # Create block manager (shared with ECC version)
    manager = SimpleBlockManager(
        num_blocks=num_blocks,
        block_size=config.block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=next(model.parameters()).device,
        codec=config.codec,  # "int4" for unprotected
    )

    # Create unprotected backend
    backend = UnprotectedBackend(manager, config, num_heads)

    original_attns = {}

    # Find rotary embedding
    rotary_emb = _find_rotary_embedding(model, layers)

    try:
        # Replace attention modules with unprotected shims
        for layer_idx, layer in enumerate(layers):
            original_attns[layer_idx] = layer.self_attn

            shim = UnprotectedPagedAttentionShim(
                original_attn=layer.self_attn,
                layer_idx=layer_idx,
                backend=backend,
                rotary_emb=rotary_emb,
            )

            layer.self_attn = shim

        # Store references for cache management
        model._unprotected_block_manager = manager
        model._unprotected_backend = backend

        yield model

    finally:
        # Restore original attention modules
        for layer_idx, original_attn in original_attns.items():
            layers[layer_idx].self_attn = original_attn

        # Clean up references
        if hasattr(model, "_unprotected_block_manager"):
            delattr(model, "_unprotected_block_manager")
        if hasattr(model, "_unprotected_backend"):
            delattr(model, "_unprotected_backend")


def reset_unprotected_cache(model):
    """Reset the unprotected KV cache for a new sequence."""
    if hasattr(model, "_unprotected_block_manager"):
        model._unprotected_block_manager.reset()


def get_unprotected_stats(model):
    """Get statistics from the unprotected backend."""
    stats = {}
    if hasattr(model, "_unprotected_block_manager"):
        manager = model._unprotected_block_manager
        stats["allocated_blocks"] = sum(
            len(blocks) for blocks in manager.seq_to_blocks.values()
        )
        stats["free_blocks"] = len(manager.free_blocks)
        stats["sequences"] = len(manager.seq_to_blocks)
    if hasattr(model, "_unprotected_backend"):
        backend = model._unprotected_backend
        stats["injection_count"] = backend._injection_count
        stats["bits_flipped"] = backend._bits_flipped
        stats["total_bits"] = backend._total_bits
        if backend._total_bits > 0:
            stats["actual_ber"] = backend._bits_flipped / backend._total_bits
        else:
            stats["actual_ber"] = 0.0
    return stats
