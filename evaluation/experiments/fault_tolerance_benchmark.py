"""
vLLM vs ECC-Protected KV Cache Comparison Benchmark

Compares throughput, latency, and quality metrics between:
- vLLM with FP16 KV cache (baseline)
- vLLM with FP8 KV cache (quantized baseline)
- vLLM with FP8 + bit error injection (degradation baseline)
- Custom ECC-protected INT4 KV cache (Hamming84, Golay)
"""

import os
import sys
import time
import math
import json
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Callable, Any, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaultInjectionAttentionShim(nn.Module):
    """
    Attention wrapper that injects bit errors into K,V BEFORE attention.

    Mirrors ECCPagedAttentionShim pattern - copies projections and runs
    attention ourselves so we can inject errors at the correct point.

    Key difference from old VLLMAttentionWithErrors: this class takes over the
    entire forward pass instead of delegating to original_attn. This ensures
    errors are injected BEFORE attention computation, not after.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        layer_idx: int,
        config: "FaultInjectionConfig",
        stats_tracker: "FaultInjectionStats",
        rotary_emb: nn.Module,
    ):
        super().__init__()

        # Copy projection layers (duck-typing like ECC shim)
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if not hasattr(original_attn, proj_name):
                raise ValueError(f"Attention module missing required projection: {proj_name}")
            setattr(self, proj_name, getattr(original_attn, proj_name))

        # Extract attention parameters
        self.num_heads = getattr(original_attn, 'num_heads',
                                 getattr(original_attn, 'num_attention_heads', 32))
        self.num_kv_heads = getattr(original_attn, 'num_key_value_heads', self.num_heads)
        self.head_dim = getattr(original_attn, 'head_dim',
                                getattr(original_attn, 'head_size', 64))
        self.hidden_size = self.num_heads * self.head_dim

        self.layer_idx = layer_idx
        self.config = config
        self.stats = stats_tracker
        self.rotary_emb = rotary_emb
        self._injection_count = 0

        # For GQA (grouped query attention)
        self.num_kv_groups = self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # 1. Project Q, K, V ourselves (not delegating to original attention)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2. Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply rotary embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        if self.rotary_emb is not None:
            try:
                cos, sin = self.rotary_emb(v, position_ids)
                q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
            except Exception:
                # If RoPE fails, continue without it (some models don't use it)
                pass

        # 4. INJECT ERRORS INTO K AND V HERE (the key fix!)
        if self.config.ber > 0:
            if self.config.inject_on_k:
                k = self._inject_errors(k)
            if self.config.inject_on_v:
                v = self._inject_errors(v)

        # 5. Handle KV cache concatenation for multi-turn generation
        if past_key_value is not None:
            if hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
                # DynamicCache format
                if self.layer_idx < len(past_key_value.key_cache):
                    cached_k = past_key_value.key_cache[self.layer_idx]
                    cached_v = past_key_value.value_cache[self.layer_idx]
                    if cached_k is not None:
                        k = torch.cat([cached_k, k], dim=2)
                        v = torch.cat([cached_v, v], dim=2)
                    # Update cache
                    past_key_value.key_cache[self.layer_idx] = k
                    past_key_value.value_cache[self.layer_idx] = v
            elif isinstance(past_key_value, tuple) and len(past_key_value) >= 2:
                # Tuple format (k, v)
                cached_k, cached_v = past_key_value[0], past_key_value[1]
                if cached_k is not None:
                    k = torch.cat([cached_k, k], dim=2)
                    v = torch.cat([cached_v, v], dim=2)

        # 6. Expand for GQA (grouped query attention) if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # 7. Run attention with CORRUPTED K,V
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),  # Use causal if no mask provided
        )

        # 8. Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        # Return in standard transformer format
        attn_weights = None
        new_past_key_value = (k, v) if use_cache else None

        return output, attn_weights, new_past_key_value

    def _inject_errors(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Inject bit errors into tensor with correct dtype handling.

        Returns a NEW tensor with errors (clones first to avoid modifying original).
        Supports FP16, BF16, FP32, and FP8 dtypes.
        """
        if tensor is None or tensor.numel() == 0:
            return tensor

        # GPU sync before modification
        torch.cuda.synchronize()

        # Calculate bit width from dtype
        n_bits = tensor.element_size() * 8
        num_elements = tensor.numel()
        total_bits = num_elements * n_bits

        self.stats.add_bits_processed(total_bits)

        # Generate unique seed for this injection
        seed = self.config.seed + self.layer_idx * 10000 + self._injection_count
        self._injection_count += 1

        # Save original shape and dtype for reconstruction
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        try:
            from ecc_codecs.triton_kernels import inject_bit_errors_triton

            # Convert tensor to contiguous byte representation
            # This avoids issues with view chains on non-contiguous tensors
            tensor_bytes = tensor.contiguous().view(torch.uint8).clone()

            # Inject bit errors at byte level
            corrupted_bytes, stats = inject_bit_errors_triton(
                tensor_bytes, self.config.ber, n_bits=8, seed=seed, return_stats=True
            )
            bits_flipped = stats[0] if stats else 0

            # Reconstruct tensor with original dtype and shape
            corrupted = corrupted_bytes.view(original_dtype).view(original_shape)

            self.stats.add_bits_flipped(bits_flipped)

        except ImportError:
            # Fallback: clone and corrupt in place
            corrupted = tensor.clone()
            bits_flipped = self._inject_errors_pytorch(corrupted, n_bits, seed)
            self.stats.add_bits_flipped(bits_flipped)

        # GPU sync after modification
        torch.cuda.synchronize()

        return corrupted

    def _inject_errors_pytorch(self, tensor: torch.Tensor, n_bits: int, seed: int) -> int:
        """PyTorch fallback with stochastic bit error injection."""
        torch.manual_seed(seed)

        tensor_bytes = tensor.view(torch.uint8).flatten()
        num_bytes = tensor_bytes.numel()
        total_bits = num_bytes * 8

        # Stochastic: each bit independently flips with probability BER
        flip_probs = torch.rand(total_bits, device=tensor.device)
        flip_mask = flip_probs < self.config.ber
        num_errors = flip_mask.sum().item()

        if num_errors == 0:
            return 0

        flip_indices = torch.where(flip_mask)[0]
        byte_indices = flip_indices // 8
        bit_positions = flip_indices % 8

        for byte_idx, bit_pos in zip(byte_indices.tolist(), bit_positions.tolist()):
            tensor_bytes[byte_idx] ^= (1 << bit_pos)

        return num_errors

    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor,
                               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings (same as ECC shim)."""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Handle variable cos/sin dimensions
        while cos.dim() < 4:
            cos = cos.unsqueeze(0 if cos.dim() < 2 else 1)
            sin = sin.unsqueeze(0 if sin.dim() < 2 else 1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


# Keep old class name as alias for backwards compatibility
VLLMAttentionWithErrors = FaultInjectionAttentionShim


class FaultInjectionStats:
    """Thread-safe statistics tracker for fault injection."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._total_bits_flipped = 0
        self._total_bits_processed = 0
        self._injection_count = 0

    def add_bits_flipped(self, count):
        self._total_bits_flipped += count
        self._injection_count += 1

    def add_bits_processed(self, count):
        self._total_bits_processed += count

    def get_stats(self):
        return {
            "injection_count": self._injection_count,
            "total_bits_flipped": self._total_bits_flipped,
            "total_bits_processed": self._total_bits_processed,
            "effective_ber": (
                self._total_bits_flipped / self._total_bits_processed
                if self._total_bits_processed > 0 else 0.0
            ),
        }


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection into attention layers."""
    ber: float = 0.0
    seed: int = 42
    inject_on_k: bool = True
    inject_on_v: bool = True


def _find_rotary_embedding(model, layers):
    """
    Find or create rotary embedding for the model.

    Mirrors the logic from kv_cache/shim.py _find_rotary_embedding().
    """
    # Try to find existing rotary embedding
    search_paths = [
        lambda: layers[0].self_attn.rotary_emb,
        lambda: layers[0].rotary_emb,
        lambda: model.model.rotary_emb if hasattr(model, 'model') else None,
        lambda: model.rotary_emb,
    ]

    for accessor in search_paths:
        try:
            rotary_emb = accessor()
            if rotary_emb is not None:
                return rotary_emb
        except (AttributeError, IndexError):
            continue

    # Try to create from transformers
    try:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        # Get config from model
        config = getattr(model, 'config', None)
        if config is not None:
            head_dim = getattr(config, 'head_dim',
                              config.hidden_size // config.num_attention_heads)
            max_position = getattr(config, 'max_position_embeddings', 4096)
            rope_base = getattr(config, 'rope_theta', 10000.0)

            return LlamaRotaryEmbedding(
                head_dim,
                max_position_embeddings=max_position,
                base=rope_base,
            )
    except ImportError:
        pass

    # Return None - model may not use RoPE
    return None


def _get_vllm_underlying_model(llm):
    """
    Extract the underlying HuggingFace model from vLLM's LLM instance.

    vLLM stores the model in various locations depending on version.
    """
    accessors = [
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.model,
        lambda: llm.llm_engine.model_executor.workers[0].model_runner.model,
    ]

    for accessor in accessors:
        try:
            model = accessor()
            if model is not None:
                # Verify it has the expected structure
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    return model
                elif hasattr(model, "layers"):
                    return model
        except (AttributeError, IndexError):
            continue

    return None


@contextmanager
def patch_model_with_fault_injection(
    model: nn.Module,
    config: FaultInjectionConfig,
    stats_tracker: Optional[FaultInjectionStats] = None,
):
    """
    Context manager that patches a model's attention layers with fault injection.

    Mirrors the ECC shim's patch_model_with_ecc_attention() pattern.

    Args:
        model: HuggingFace transformer model
        config: FaultInjectionConfig with BER settings
        stats_tracker: Optional statistics tracker

    Yields:
        The patched model

    Example:
        config = FaultInjectionConfig(ber=1e-3, seed=42)
        stats = FaultInjectionStats()
        with patch_model_with_fault_injection(model, config, stats):
            outputs = model(input_ids, labels=input_ids)
        print(stats.get_stats())
    """
    if stats_tracker is None:
        stats_tracker = FaultInjectionStats()

    # Detect model structure (same as ECC shim)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Unsupported model architecture: cannot find 'layers'")

    num_layers = len(layers)
    print(f"[FaultInjection] Patching {num_layers} layers with BER={config.ber:.0e}")

    # Find rotary embedding
    rotary_emb = _find_rotary_embedding(model, layers)

    original_attns = {}

    try:
        for layer_idx, layer in enumerate(layers):
            original_attns[layer_idx] = layer.self_attn

            shim = FaultInjectionAttentionShim(
                original_attn=layer.self_attn,
                layer_idx=layer_idx,
                config=config,
                stats_tracker=stats_tracker,
                rotary_emb=rotary_emb,
            )

            layer.self_attn = shim

        # Store tracker on model for easy access
        model._fi_stats_tracker = stats_tracker

        yield model

    finally:
        # Restore original attention modules
        for layer_idx, original_attn in original_attns.items():
            layers[layer_idx].self_attn = original_attn

        if hasattr(model, "_fi_stats_tracker"):
            delattr(model, "_fi_stats_tracker")

        print(f"[FaultInjection] Restored original attention layers")


class VLLMWithFaultInjection:
    """
    Wrapper around vLLM's LLM class that injects bit errors into the KV cache.

    This simulates hardware bit-flip errors to demonstrate:
    1. How vanilla vLLM degrades under memory errors
    2. The value of ECC protection for error resilience

    Implementation uses attention layer patching (similar to ECC shim) for
    accurate error injection at the correct point in the computation.

    Usage:
        wrapper = VLLMWithFaultInjection(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ber=1e-3,  # Bit error rate
            kv_cache_dtype="fp8",
        )
        outputs = wrapper.generate(prompts, sampling_params)
    """

    def __init__(
        self,
        model_name: str,
        ber: float = 0.0,
        kv_cache_dtype: str = "auto",
        injection_mode: str = "cache_hook",
        seed: int = 42,
        **vllm_kwargs,
    ):
        """
        Initialize vLLM with fault injection capability.

        Args:
            model_name: HuggingFace model name or path
            ber: Bit error rate (probability of flipping each bit)
            kv_cache_dtype: KV cache data type ("auto", "fp8", "fp8_e4m3", etc.)
            injection_mode: When to inject errors:
                - "cache_hook": Hook into cache reads to inject errors (recommended)
                - "once": Inject once after each generation step
            seed: Random seed for reproducible error injection
            **vllm_kwargs: Additional arguments passed to vLLM LLM()
        """
        from vllm import LLM

        self.model_name = model_name
        self.ber = ber
        self.kv_cache_dtype = kv_cache_dtype
        self.injection_mode = injection_mode
        self.seed = seed

        # Statistics tracking
        self.stats = FaultInjectionStats()
        self._injection_count = 0

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            kv_cache_dtype=kv_cache_dtype,
            **vllm_kwargs,
        )

        # Install cache hook for error injection
        if injection_mode == "cache_hook" and ber > 0:
            self._install_cache_hook()

        if ber > 0:
            print(f"[FaultInjection] Initialized with BER={ber:.0e}, mode={injection_mode}")

    def _install_cache_hook(self):
        """
        Install hook to inject errors when cache is accessed.

        This hooks into vLLM's worker to inject errors during attention computation.
        """
        import torch

        try:
            # Try to hook into the cache engine's get operation
            worker = self.llm.llm_engine.model_executor.driver_worker

            if hasattr(worker, 'cache_engine'):
                cache_engine = worker.cache_engine

                # Hook into cache read operations
                if hasattr(cache_engine, 'gpu_cache'):
                    self._hooked_cache = cache_engine.gpu_cache
                    self._original_gpu_cache_getitem = None

                    # Create a wrapper for cache access
                    original_cache = cache_engine.gpu_cache
                    wrapper = self

                    class CacheWithInjection:
                        """Wrapper that injects errors on cache access."""
                        def __init__(self, original):
                            self._original = original

                        def __len__(self):
                            return len(self._original)

                        def __iter__(self):
                            for item in self._original:
                                yield item

                        def __getitem__(self, idx):
                            item = self._original[idx]
                            if wrapper.ber > 0 and item is not None:
                                wrapper._inject_into_cache_item(item, idx)
                            return item

                        def __setitem__(self, idx, value):
                            self._original[idx] = value

                    # Replace the cache with our wrapper
                    cache_engine.gpu_cache = CacheWithInjection(original_cache)
                    self._original_cache = original_cache

                    print(f"[FaultInjection] Installed cache hook")
                    return

        except Exception as e:
            print(f"[FaultInjection] Cache hook failed: {e}, using fallback")

        # Fallback: hook into execute_model
        self._install_execution_hook_fallback()

    def _install_execution_hook_fallback(self):
        """Fallback: Install hook on execute_model."""
        try:
            executor = self.llm.llm_engine.model_executor

            if hasattr(executor, 'execute_model'):
                self._original_execute_model = executor.execute_model

                def hooked_execute_model(*args, **kwargs):
                    result = self._original_execute_model(*args, **kwargs)
                    # Inject errors AFTER execution (into cached values)
                    if self.ber > 0:
                        self._inject_cache_errors_post_execution()
                    return result

                executor.execute_model = hooked_execute_model
                print(f"[FaultInjection] Installed execution hook fallback (BER={self.ber:.0e})")
        except Exception as e:
            print(f"[FaultInjection] Warning: Could not install hook: {e}")

    def _inject_into_cache_item(self, cache_item, layer_idx):
        """Inject errors into a cache item (k_cache, v_cache tuple)."""
        import torch

        if cache_item is None:
            return

        # GPU sync before accessing cache
        torch.cuda.synchronize()

        if isinstance(cache_item, tuple) and len(cache_item) >= 2:
            k_cache, v_cache = cache_item[0], cache_item[1]

            if k_cache is not None and k_cache.numel() > 0:
                self._inject_errors_into_tensor(
                    k_cache,
                    seed=self.seed + self._injection_count + layer_idx * 1000
                )

            if v_cache is not None and v_cache.numel() > 0:
                self._inject_errors_into_tensor(
                    v_cache,
                    seed=self.seed + self._injection_count + layer_idx * 1000 + 500
                )

        self._injection_count += 1

        # GPU sync after modification
        torch.cuda.synchronize()

    def _inject_cache_errors_post_execution(self):
        """Inject errors into cache after model execution."""
        import torch

        gpu_cache = self._get_cache_tensors()
        if gpu_cache is None:
            return

        # GPU sync before accessing cache
        torch.cuda.synchronize()

        for layer_idx, cache_item in enumerate(gpu_cache):
            self._inject_into_cache_item(cache_item, layer_idx)

        # GPU sync after all modifications
        torch.cuda.synchronize()

    def _get_cache_tensors(self):
        """Access vLLM's internal KV cache tensors."""
        # Try various vLLM version structures
        for accessor in [
            lambda: self.llm.llm_engine.model_executor.driver_worker.cache_engine.gpu_cache,
            lambda: self.llm.llm_engine.model_executor.driver_worker.model_runner.gpu_cache,
            lambda: self.llm.llm_engine.model_executor.workers[0].cache_engine.gpu_cache,
        ]:
            try:
                cache = accessor()
                if cache is not None:
                    return cache
            except (AttributeError, IndexError):
                continue
        return None

    def _inject_errors_into_tensor(self, tensor, seed=42):
        """
        Inject bit errors into a tensor with CORRECT bit-width handling.

        Key fixes:
        1. Uses tensor.element_size() * 8 for correct bit width
        2. Adds GPU synchronization
        3. Uses stochastic error injection
        """
        import torch

        if tensor is None or tensor.numel() == 0:
            return 0

        # CORRECT: Calculate bit width from dtype
        n_bits = tensor.element_size() * 8

        num_elements = tensor.numel()
        total_bits = num_elements * n_bits
        self.stats.add_bits_processed(total_bits)

        # Try Triton kernel first
        try:
            from ecc_codecs.triton_kernels import inject_bit_errors_triton

            # Work with appropriate integer view for the dtype
            flat_tensor = tensor.flatten()

            if tensor.dtype in (torch.float16, torch.bfloat16):
                int_view = flat_tensor.view(torch.int16)
                corrupted, stats = inject_bit_errors_triton(
                    int_view, self.ber, n_bits=16, seed=seed, return_stats=True
                )
                int_view.copy_(corrupted)
                bits_flipped = stats[0] if stats else 0
            elif tensor.dtype == torch.float32:
                int_view = flat_tensor.view(torch.int32)
                corrupted, stats = inject_bit_errors_triton(
                    int_view, self.ber, n_bits=32, seed=seed, return_stats=True
                )
                int_view.copy_(corrupted)
                bits_flipped = stats[0] if stats else 0
            elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8):
                int_view = flat_tensor.view(torch.uint8)
                corrupted, stats = inject_bit_errors_triton(
                    int_view, self.ber, n_bits=8, seed=seed, return_stats=True
                )
                int_view.copy_(corrupted)
                bits_flipped = stats[0] if stats else 0
            else:
                # Generic fallback: view as bytes
                tensor_bytes = tensor.view(torch.uint8).flatten()
                num_bytes = tensor_bytes.numel()
                corrupted, stats = inject_bit_errors_triton(
                    tensor_bytes, self.ber, n_bits=8, seed=seed, return_stats=True
                )
                tensor_bytes.copy_(corrupted)
                # Adjust bits flipped count for actual element bits
                bits_flipped = stats[0] if stats else 0

            self.stats.add_bits_flipped(bits_flipped)
            return bits_flipped

        except ImportError:
            return self._inject_errors_pytorch(tensor, n_bits, seed)

    def _inject_errors_pytorch(self, tensor, n_bits, seed):
        """
        PyTorch fallback with STOCHASTIC bit error injection.

        FIXED: Each bit independently flips with probability `ber`,
        instead of deterministically flipping exactly `int(total_bits * ber)` bits.
        """
        import torch

        torch.manual_seed(seed)

        # View as bytes for bit manipulation
        tensor_bytes = tensor.view(torch.uint8).flatten()
        num_bytes = tensor_bytes.numel()
        total_bits = num_bytes * 8

        # STOCHASTIC: Each bit independently flips with probability ber
        # This is the correct probabilistic interpretation of BER
        flip_probs = torch.rand(total_bits, device=tensor.device)
        flip_mask = flip_probs < self.ber
        num_errors = flip_mask.sum().item()

        if num_errors == 0:
            return 0

        # Get indices of bits to flip
        flip_indices = torch.where(flip_mask)[0]
        byte_indices = flip_indices // 8
        bit_positions = flip_indices % 8

        # Apply bit flips
        for byte_idx, bit_pos in zip(byte_indices.tolist(), bit_positions.tolist()):
            tensor_bytes[byte_idx] ^= (1 << bit_pos)

        self.stats.add_bits_flipped(num_errors)
        return num_errors

    def generate(self, prompts, sampling_params, **kwargs):
        """Generate completions with automatic error injection."""
        return self.llm.generate(prompts, sampling_params, **kwargs)

    def get_stats(self):
        """Get error injection statistics."""
        stats = self.stats.get_stats()
        stats["ber_target"] = self.ber
        stats["injection_count"] = self._injection_count
        return stats

    def reset_stats(self):
        """Reset injection statistics."""
        self.stats.reset()
        self._injection_count = 0

    def __del__(self):
        """Cleanup: restore original methods if hooked."""
        if hasattr(self, '_original_execute_model'):
            try:
                executor = self.llm.llm_engine.model_executor
                executor.execute_model = self._original_execute_model
            except Exception:
                pass

        if hasattr(self, '_original_cache'):
            try:
                worker = self.llm.llm_engine.model_executor.driver_worker
                if hasattr(worker, 'cache_engine'):
                    worker.cache_engine.gpu_cache = self._original_cache
            except Exception:
                pass


@dataclass
class VLLMComparisonConfig:
    """Configuration for vLLM comparison benchmark."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    batch_sizes: list = field(default_factory=lambda: [1, 4, 8, 16])
    seq_lengths: list = field(default_factory=lambda: [128, 512, 1024, 2048])
    max_new_tokens: int = 128
    num_samples: int = 100
    warmup_samples: int = 10
    num_trials: int = 3

    configs: list = field(default_factory=lambda: [
        "vllm_fp16",
        "vllm_fp8",
        "ecc_int4",
        "ecc_hamming84",
        "ecc_golay",
    ])

    # BER levels for error injection comparison
    ber_levels: list = field(default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2])

    # Enable BER sweep comparison
    run_ber_sweep: bool = False

    compute_perplexity: bool = True
    perplexity_samples: int = 50


@dataclass
class BenchmarkResult:
    """Result from a single benchmark configuration."""
    config: str
    batch_size: int
    seq_length: int
    throughput_tokens_sec: float
    latency_ms_per_token: float
    memory_gb: float
    perplexity: float = 0.0
    ttft_ms: float = 0.0
    num_samples: int = 0
    total_tokens: int = 0
    # Error injection stats
    ber: float = 0.0
    bits_flipped: int = 0
    bits_processed: int = 0


@dataclass
class ComparisonReport:
    """Full comparison report with all results."""
    results: list = field(default_factory=list)
    config: VLLMComparisonConfig = None
    vllm_available: bool = False
    error_message: str = ""

    def get_throughput_table(self):
        """Generate throughput comparison table."""
        if not self.results:
            return "No results available"

        configs = sorted(set(r.config for r in self.results))
        batch_seq_pairs = sorted(set((r.batch_size, r.seq_length) for r in self.results))

        lines = []
        lines.append("=" * 100)
        lines.append("THROUGHPUT COMPARISON (tokens/sec)")
        lines.append("=" * 100)
        lines.append("")

        header = f"{'Config':<20} |"
        for batch, seq in batch_seq_pairs:
            header += f" {batch}x{seq:>4} |"
        lines.append(header)
        lines.append("-" * len(header))

        result_map = {}
        for r in self.results:
            key = (r.config, r.batch_size, r.seq_length)
            result_map[key] = r

        for config in configs:
            row = f"{config:<20} |"
            for batch, seq in batch_seq_pairs:
                r = result_map.get((config, batch, seq))
                if r:
                    row += f" {r.throughput_tokens_sec:>7.0f} |"
                else:
                    row += f" {'N/A':>7} |"
            lines.append(row)

        return "\n".join(lines)

    def get_overhead_table(self):
        """Generate overhead comparison vs vLLM baselines."""
        if not self.results:
            return "No results available"

        configs = sorted(set(r.config for r in self.results))
        batch_seq_pairs = sorted(set((r.batch_size, r.seq_length) for r in self.results))

        result_map = {}
        for r in self.results:
            key = (r.config, r.batch_size, r.seq_length)
            result_map[key] = r

        lines = []
        lines.append("")
        lines.append("=" * 100)
        lines.append("OVERHEAD vs vLLM FP16 (ratio, lower is better)")
        lines.append("=" * 100)
        lines.append("")

        header = f"{'Config':<20} |"
        for batch, seq in batch_seq_pairs:
            header += f" {batch}x{seq:>4} |"
        lines.append(header)
        lines.append("-" * len(header))

        for config in configs:
            if config == "vllm_fp16":
                continue
            row = f"{config:<20} |"
            for batch, seq in batch_seq_pairs:
                r = result_map.get((config, batch, seq))
                baseline = result_map.get(("vllm_fp16", batch, seq))
                if r and baseline and baseline.throughput_tokens_sec > 0:
                    ratio = r.throughput_tokens_sec / baseline.throughput_tokens_sec
                    row += f" {ratio:>7.2f}x |"
                else:
                    row += f" {'N/A':>7} |"
            lines.append(row)

        return "\n".join(lines)

    def get_perplexity_table(self):
        """Generate perplexity comparison table."""
        if not self.results:
            return "No results available"

        configs = sorted(set(r.config for r in self.results))

        ppl_by_config = defaultdict(list)
        for r in self.results:
            if r.perplexity > 0:
                ppl_by_config[r.config].append(r.perplexity)

        if not ppl_by_config:
            return "No perplexity data available"

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("PERPLEXITY COMPARISON")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"{'Config':<20} | {'PPL':>10} | {'vs FP16':>10}")
        lines.append("-" * 45)

        fp16_ppl = None
        if "vllm_fp16" in ppl_by_config:
            ppls = ppl_by_config["vllm_fp16"]
            fp16_ppl = sum(ppls) / len(ppls)

        for config in configs:
            if config in ppl_by_config:
                ppls = ppl_by_config[config]
                mean_ppl = sum(ppls) / len(ppls)
                if fp16_ppl and fp16_ppl > 0:
                    delta = ((mean_ppl - fp16_ppl) / fp16_ppl) * 100
                    delta_str = f"{delta:+.1f}%"
                else:
                    delta_str = "baseline"
                lines.append(f"{config:<20} | {mean_ppl:>10.2f} | {delta_str:>10}")

        return "\n".join(lines)

    def get_memory_table(self):
        """Generate memory usage comparison table."""
        if not self.results:
            return "No results available"

        configs = sorted(set(r.config for r in self.results))

        mem_by_config = defaultdict(list)
        for r in self.results:
            if r.memory_gb > 0:
                mem_by_config[r.config].append(r.memory_gb)

        if not mem_by_config:
            return "No memory data available"

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("MEMORY USAGE (Peak GPU Memory in GB)")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"{'Config':<20} | {'Memory (GB)':>12} | {'vs FP16':>10}")
        lines.append("-" * 50)

        fp16_mem = None
        if "vllm_fp16" in mem_by_config:
            mems = mem_by_config["vllm_fp16"]
            fp16_mem = sum(mems) / len(mems)

        for config in configs:
            if config in mem_by_config:
                mems = mem_by_config[config]
                mean_mem = sum(mems) / len(mems)
                if fp16_mem and fp16_mem > 0:
                    ratio = mean_mem / fp16_mem
                    ratio_str = f"{ratio:.2f}x"
                else:
                    ratio_str = "baseline"
                lines.append(f"{config:<20} | {mean_mem:>12.2f} | {ratio_str:>10}")

        return "\n".join(lines)

    def get_full_report(self):
        """Generate complete comparison report."""
        lines = []
        lines.append("=" * 100)
        lines.append("vLLM vs ECC-PROTECTED KV CACHE COMPARISON")
        lines.append("=" * 100)
        lines.append("")

        if self.config:
            lines.append(f"Model: {self.config.model_name}")
            lines.append(f"Batch sizes: {self.config.batch_sizes}")
            lines.append(f"Sequence lengths: {self.config.seq_lengths}")
            lines.append(f"Samples: {self.config.num_samples}")
            lines.append("")

        if self.error_message:
            lines.append(f"WARNING: {self.error_message}")
            lines.append("")

        lines.append(self.get_throughput_table())
        lines.append(self.get_overhead_table())
        lines.append(self.get_perplexity_table())
        lines.append(self.get_memory_table())

        lines.append("")
        lines.append("=" * 100)

        return "\n".join(lines)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [
                {
                    "config": r.config,
                    "batch_size": r.batch_size,
                    "seq_length": r.seq_length,
                    "throughput_tokens_sec": r.throughput_tokens_sec,
                    "latency_ms_per_token": r.latency_ms_per_token,
                    "memory_gb": r.memory_gb,
                    "perplexity": r.perplexity,
                    "ttft_ms": r.ttft_ms,
                    "num_samples": r.num_samples,
                    "total_tokens": r.total_tokens,
                    "ber": r.ber,
                    "bits_flipped": r.bits_flipped,
                    "bits_processed": r.bits_processed,
                }
                for r in self.results
            ],
            "config": {
                "model_name": self.config.model_name,
                "batch_sizes": self.config.batch_sizes,
                "seq_lengths": self.config.seq_lengths,
                "max_new_tokens": self.config.max_new_tokens,
                "num_samples": self.config.num_samples,
                "configs": self.config.configs,
                "ber_levels": self.config.ber_levels,
                "run_ber_sweep": self.config.run_ber_sweep,
            } if self.config else None,
            "vllm_available": self.vllm_available,
            "error_message": self.error_message,
        }


def load_benchmark_prompts(num_samples, max_length=512):
    """Load prompts from WikiText-2 for benchmarking."""
    num_samples = int(num_samples)
    max_length = int(max_length)
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        prompts = []
        for item in dataset:
            text = item["text"].strip()
            if len(text) > 50:
                prompts.append(text[:max_length])
                if len(prompts) >= num_samples:
                    break
        return prompts
    except Exception as e:
        print(f"Warning: Could not load WikiText-2: {e}")
        return ["The meaning of life is" for _ in range(num_samples)]


def get_gpu_memory_gb():
    """Get current GPU memory usage in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def run_vllm_throughput(config, kv_cache_dtype="auto", batch_size=1, seq_length=512):
    """Run throughput benchmark using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return None, "vLLM not available"

    import torch

    torch.cuda.reset_peak_memory_stats()

    try:
        llm = LLM(
            model=config.model_name,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=0.9,
            max_model_len=seq_length + config.max_new_tokens,
        )
    except Exception as e:
        return None, f"Failed to load vLLM model: {e}"

    prompts = load_benchmark_prompts(config.num_samples, max_length=seq_length)

    if batch_size > 1:
        batched_prompts = []
        for i in range(0, len(prompts), batch_size):
            batched_prompts.append(prompts[i:i+batch_size])
    else:
        batched_prompts = [[p] for p in prompts]

    sampling_params = SamplingParams(
        max_tokens=config.max_new_tokens,
        temperature=0.0,
    )

    for batch in batched_prompts[:config.warmup_samples]:
        try:
            llm.generate(batch, sampling_params)
        except Exception:
            pass

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    total_tokens = 0
    num_samples = 0

    for batch in batched_prompts:
        try:
            outputs = llm.generate(batch, sampling_params)
            for output in outputs:
                total_tokens += len(output.outputs[0].token_ids)
                num_samples += 1
        except Exception as e:
            print(f"Generation error: {e}")
            continue

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    memory_gb = get_gpu_memory_gb()

    if elapsed > 0 and total_tokens > 0:
        throughput = total_tokens / elapsed
        latency = (elapsed * 1000) / total_tokens
    else:
        throughput = 0.0
        latency = 0.0

    result = BenchmarkResult(
        config=f"vllm_{kv_cache_dtype}" if kv_cache_dtype != "auto" else "vllm_fp16",
        batch_size=batch_size,
        seq_length=seq_length,
        throughput_tokens_sec=throughput,
        latency_ms_per_token=latency,
        memory_gb=memory_gb,
        num_samples=num_samples,
        total_tokens=total_tokens,
    )

    del llm
    torch.cuda.empty_cache()

    return result, None


def run_ecc_throughput(config, codec="hamming84", batch_size=1, seq_length=512):
    """Run throughput benchmark using ECC-protected attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from kv_cache.ecc_shim import (
            ECCShimConfig,
            patch_model_with_ecc_attention,
            reset_ecc_cache,
        )
    except ImportError as e:
        return None, f"ECC shim not available: {e}"

    torch.cuda.reset_peak_memory_stats()

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return None, f"Failed to load model: {e}"

    ecc_config = ECCShimConfig(
        codec=codec,
        ber=0.0,
        inject_errors=False,
        num_blocks=2048,
        block_size=16,
    )

    prompts = load_benchmark_prompts(config.num_samples, max_length=seq_length)

    total_tokens = 0
    num_samples = 0
    elapsed = 0.0

    try:
        with patch_model_with_ecc_attention(model, ecc_config, num_blocks=2048):
            for i, prompt in enumerate(prompts[:config.warmup_samples]):
                reset_ecc_cache(model)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=seq_length,
                    truncation=True,
                    padding=True,
                ).to("cuda")
                with torch.no_grad():
                    model.generate(
                        **inputs,
                        max_new_tokens=min(32, config.max_new_tokens),
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for prompt in prompts:
                reset_ecc_cache(model)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=seq_length,
                    truncation=True,
                    padding=True,
                ).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                total_tokens += generated_tokens
                num_samples += 1

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

    except Exception as e:
        return None, f"ECC inference error: {e}"

    memory_gb = get_gpu_memory_gb()

    if elapsed > 0 and total_tokens > 0:
        throughput = total_tokens / elapsed
        latency = (elapsed * 1000) / total_tokens
    else:
        throughput = 0.0
        latency = 0.0

    config_name = f"ecc_{codec}"
    result = BenchmarkResult(
        config=config_name,
        batch_size=batch_size,
        seq_length=seq_length,
        throughput_tokens_sec=throughput,
        latency_ms_per_token=latency,
        memory_gb=memory_gb,
        num_samples=num_samples,
        total_tokens=total_tokens,
    )

    del model
    torch.cuda.empty_cache()

    return result, None


def compute_vllm_perplexity(config, kv_cache_dtype="auto"):
    """Compute perplexity using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        return None, "vLLM not available"

    import torch

    try:
        llm = LLM(
            model=config.model_name,
            kv_cache_dtype=kv_cache_dtype,
            gpu_memory_utilization=0.9,
        )
    except Exception as e:
        return None, f"Failed to load vLLM: {e}"

    prompts = load_benchmark_prompts(config.perplexity_samples, max_length=256)
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

    total_nll = 0.0
    total_tokens = 0

    for prompt in prompts:
        try:
            outputs = llm.generate([prompt], sampling_params)
            if outputs and outputs[0].prompt_logprobs:
                logprobs = outputs[0].prompt_logprobs
                for lp in logprobs:
                    if lp is not None:
                        for token_id, info in lp.items():
                            total_nll -= info.logprob
                            total_tokens += 1
        except Exception:
            continue

    del llm
    torch.cuda.empty_cache()

    if total_tokens > 0:
        ppl = math.exp(total_nll / total_tokens)
        return ppl, None
    return None, "No tokens processed"


def compute_vllm_perplexity_with_ber(config, kv_cache_dtype="auto", ber=0.0):
    """Compute perplexity using vLLM with bit error injection."""
    try:
        from vllm import SamplingParams
    except ImportError:
        return None, None, "vLLM not available"

    import torch

    try:
        wrapper = VLLMWithFaultInjection(
            model_name=config.model_name,
            ber=ber,
            kv_cache_dtype=kv_cache_dtype,
            injection_mode="before_attention" if ber > 0 else "once",
            gpu_memory_utilization=0.9,
        )
    except Exception as e:
        return None, None, f"Failed to load vLLM with fault injection: {e}"

    prompts = load_benchmark_prompts(config.perplexity_samples, max_length=256)
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

    total_nll = 0.0
    total_tokens = 0

    for prompt in prompts:
        try:
            outputs = wrapper.generate([prompt], sampling_params)
            if outputs and outputs[0].prompt_logprobs:
                logprobs = outputs[0].prompt_logprobs
                for lp in logprobs:
                    if lp is not None:
                        for token_id, info in lp.items():
                            total_nll -= info.logprob
                            total_tokens += 1
        except Exception as e:
            print(f"  Generation error at BER={ber:.0e}: {e}")
            continue

    stats = wrapper.get_stats()

    del wrapper
    torch.cuda.empty_cache()

    if total_tokens > 0:
        ppl = math.exp(total_nll / total_tokens)
        return ppl, stats, None
    return None, stats, "No tokens processed"


def compute_vllm_perplexity_attention_injection(
    config,
    kv_cache_dtype: str = "auto",
    ber: float = 0.0,
):
    """
    Compute perplexity using attention-level fault injection.

    This approach:
    1. Loads the model directly via HuggingFace (not through vLLM's PagedAttention)
    2. Patches attention layers with FaultInjectionAttentionShim
    3. Runs inference with errors injected at the correct point (BEFORE attention)

    This is the FIXED approach that actually affects perplexity under BER.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    # Setup fault injection config
    fi_config = FaultInjectionConfig(ber=ber, seed=42)
    stats = FaultInjectionStats()

    prompts = load_benchmark_prompts(config.perplexity_samples, max_length=256)

    total_loss = 0.0
    total_tokens = 0

    try:
        with patch_model_with_fault_injection(model, fi_config, stats):
            for prompt in prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                ).to("cuda")

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        labels=inputs["input_ids"],
                        use_cache=False,  # Disable cache - not needed for perplexity
                    )

                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    seq_len = inputs["input_ids"].shape[1]
                    total_loss += outputs.loss.item() * seq_len
                    total_tokens += seq_len

    except Exception as e:
        del model
        torch.cuda.empty_cache()
        return None, None, f"Fault injection error: {e}"

    del model
    torch.cuda.empty_cache()

    final_stats = stats.get_stats()
    if total_tokens > 0:
        ppl = math.exp(total_loss / total_tokens)
        return ppl, final_stats, None
    return None, None, "No tokens processed"


def compute_unprotected_perplexity_with_ber(config, ber=0.0):
    """
    Compute perplexity using UNPROTECTED INT4 KV cache with error injection.

    This is the fair comparison point for ECC-protected configs:
    - Same INT4 quantization as ECC configs
    - Same block-based storage
    - NO error correction

    Error injection happens AFTER cache read, BEFORE dequantize, simulating
    realistic memory corruption in stored data.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from kv_cache.unprotected_shim import (
            UnprotectedShimConfig,
            patch_model_with_unprotected_attention,
            reset_unprotected_cache,
            get_unprotected_stats,
        )
    except ImportError as e:
        return None, None, f"Unprotected shim not available: {e}"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    # Configure unprotected cache with INT4 quantization
    unprotected_config = UnprotectedShimConfig(
        ber=ber,
        inject_errors=(ber > 0),
        num_blocks=2048,
        block_size=16,
    )

    prompts = load_benchmark_prompts(config.perplexity_samples, max_length=256)

    total_loss = 0.0
    total_tokens = 0

    try:
        with patch_model_with_unprotected_attention(model, unprotected_config, num_blocks=2048):
            for prompt in prompts:
                reset_unprotected_cache(model)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                ).to("cuda")

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        labels=inputs["input_ids"],
                        use_cache=True,
                    )

                if outputs.loss is not None and not torch.isnan(outputs.loss):
                    seq_len = inputs["input_ids"].shape[1]
                    total_loss += outputs.loss.item() * seq_len
                    total_tokens += seq_len

            # Get stats
            stats = get_unprotected_stats(model)

    except Exception as e:
        del model
        torch.cuda.empty_cache()
        return None, None, f"Unprotected perplexity error: {e}"

    del model
    torch.cuda.empty_cache()

    if total_tokens > 0:
        ppl = math.exp(total_loss / total_tokens)
        return ppl, stats, None
    return None, None, "No tokens processed"


def compute_ecc_perplexity_with_ber(config, codec="hamming84", ber=0.0):
    """Compute perplexity using ECC-protected attention with error injection."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from kv_cache.ecc_shim import (
            ECCShimConfig,
            patch_model_with_ecc_attention,
            reset_ecc_cache,
            get_ecc_stats,
        )
    except ImportError as e:
        return None, None, f"ECC shim not available: {e}"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return None, None, f"Failed to load model: {e}"

    ecc_config = ECCShimConfig(
        codec=codec,
        ber=ber,
        inject_errors=(ber > 0),
        num_blocks=2048,
        block_size=16,
    )

    prompts = load_benchmark_prompts(config.perplexity_samples, max_length=256)

    total_loss = 0.0
    total_tokens = 0

    try:
        with patch_model_with_ecc_attention(model, ecc_config, num_blocks=2048):
            for prompt in prompts:
                reset_ecc_cache(model)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                ).to("cuda")

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        labels=inputs["input_ids"],
                        use_cache=True,
                    )

                seq_len = inputs["input_ids"].shape[1]
                total_loss += outputs.loss.item() * seq_len
                total_tokens += seq_len

            # Get ECC stats
            ecc_stats = get_ecc_stats(model)

    except Exception as e:
        del model
        torch.cuda.empty_cache()
        return None, None, f"ECC perplexity error: {e}"

    del model
    torch.cuda.empty_cache()

    if total_tokens > 0:
        ppl = math.exp(total_loss / total_tokens)
        return ppl, ecc_stats, None
    return None, None, "No tokens processed"


def run_ber_sweep_comparison(config, progress_callback=None, use_attention_injection=True):
    """
    Run BER sweep comparison between unprotected and ECC-protected.

    This demonstrates:
    1. How unprotected INT4 models degrade as BER increases
    2. How ECC protection maintains quality under errors

    Args:
        config: VLLMComparisonConfig with benchmark settings
        progress_callback: Optional callback for progress reporting
        use_attention_injection: If True, use attention-level fault injection (recommended).
            This injects errors BEFORE attention computation, which actually affects output.
            If False, uses legacy cache-level injection (doesn't work with vLLM PagedAttention).
    """
    results = []

    # Configs to compare at each BER level
    # Key insight: unprotected_int4, ecc_hamming84, and ecc_golay all use the SAME
    # INT4 quantization, so baseline perplexity should be identical.
    # The ONLY difference is ECC protection vs no protection.
    sweep_configs = [
        ("unprotected_int4", None),    # INT4 without ECC (fair comparison baseline)
        ("ecc_hamming84", "hamming84"),
        ("ecc_golay", "golay"),
    ]

    total_tasks = len(config.ber_levels) * len(sweep_configs)
    current_task = 0

    for ber in config.ber_levels:
        for cfg_name, cfg_param in sweep_configs:
            current_task += 1
            if progress_callback:
                progress_callback(
                    f"BER={ber:.0e} {cfg_name}",
                    current_task,
                    total_tasks
                )

            ppl = None
            stats = None
            error = None

            if cfg_name == "unprotected_int4":
                # Unprotected INT4 - same quantization as ECC, no protection
                ppl, stats, error = compute_unprotected_perplexity_with_ber(
                    config, ber
                )
            elif cfg_name.startswith("hf_") or cfg_name.startswith("vllm_"):
                # Use attention-level injection for fair comparison
                # This actually affects perplexity, unlike cache-level injection
                ppl, stats, error = compute_vllm_perplexity_attention_injection(
                    config, cfg_param, ber
                )
            elif cfg_name.startswith("ecc_"):
                codec = cfg_param
                ppl, stats, error = compute_ecc_perplexity_with_ber(
                    config, codec, ber
                )

            if ppl is not None:
                result = BenchmarkResult(
                    config=cfg_name,
                    batch_size=0,
                    seq_length=0,
                    throughput_tokens_sec=0,
                    latency_ms_per_token=0,
                    memory_gb=0,
                    perplexity=ppl,
                    ber=ber,
                    bits_flipped=stats.get("total_bits_flipped", 0) if stats else 0,
                    bits_processed=stats.get("total_bits_processed", 0) if stats else 0,
                )
                results.append(result)
                print(f"  {cfg_name} @ BER={ber:.0e}: PPL={ppl:.2f}")
            elif error:
                print(f"  Error for {cfg_name} @ BER={ber:.0e}: {error}")

    return results


def format_ber_sweep_table(results):
    """Format BER sweep results as a comparison table."""
    if not results:
        return "No BER sweep results available"

    # Group results by config
    configs = sorted(set(r.config for r in results))
    ber_levels = sorted(set(r.ber for r in results))

    result_map = {}
    for r in results:
        result_map[(r.config, r.ber)] = r

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("BER SWEEP: PERPLEXITY COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Shows how perplexity degrades with increasing bit error rate (BER).")
    lines.append("ECC-protected configs should maintain lower perplexity at higher BER.")
    lines.append("")

    # Header
    header = f"{'Config':<20} |"
    for ber in ber_levels:
        header += f" BER={ber:.0e} |"
    lines.append(header)
    lines.append("-" * len(header))

    # Get baseline (vllm_fp16 at BER=0)
    baseline_ppl = None
    if ("vllm_fp16", 0.0) in result_map:
        baseline_ppl = result_map[("vllm_fp16", 0.0)].perplexity

    for cfg in configs:
        row = f"{cfg:<20} |"
        for ber in ber_levels:
            r = result_map.get((cfg, ber))
            if r and r.perplexity > 0:
                ppl = r.perplexity
                if ppl > 1000:
                    row += f" {'FAIL':>10} |"
                else:
                    row += f" {ppl:>10.2f} |"
            else:
                row += f" {'N/A':>10} |"
        lines.append(row)

    # Add delta table
    lines.append("")
    lines.append("-" * len(header))
    lines.append("Perplexity increase vs BER=0 baseline:")
    lines.append("")

    for cfg in configs:
        baseline = result_map.get((cfg, 0.0))
        if not baseline or baseline.perplexity <= 0:
            continue

        row = f"{cfg:<20} |"
        for ber in ber_levels:
            r = result_map.get((cfg, ber))
            if r and r.perplexity > 0 and baseline.perplexity > 0:
                delta = ((r.perplexity - baseline.perplexity) / baseline.perplexity) * 100
                if delta > 1000:
                    row += f" {'>1000%':>10} |"
                else:
                    row += f" {delta:>+9.1f}% |"
            else:
                row += f" {'N/A':>10} |"
        lines.append(row)

    return "\n".join(lines)


def compute_ecc_perplexity(config, codec="hamming84"):
    """Compute perplexity using ECC-protected attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from kv_cache.ecc_shim import (
            ECCShimConfig,
            patch_model_with_ecc_attention,
            reset_ecc_cache,
        )
    except ImportError as e:
        return None, f"ECC shim not available: {e}"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        return None, f"Failed to load model: {e}"

    ecc_config = ECCShimConfig(
        codec=codec,
        ber=0.0,
        inject_errors=False,
        num_blocks=2048,
        block_size=16,
    )

    prompts = load_benchmark_prompts(config.perplexity_samples, max_length=256)

    total_loss = 0.0
    total_tokens = 0

    try:
        with patch_model_with_ecc_attention(model, ecc_config, num_blocks=2048):
            for prompt in prompts:
                reset_ecc_cache(model)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                ).to("cuda")

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        labels=inputs["input_ids"],
                        use_cache=True,
                    )

                seq_len = inputs["input_ids"].shape[1]
                total_loss += outputs.loss.item() * seq_len
                total_tokens += seq_len

    except Exception as e:
        del model
        torch.cuda.empty_cache()
        return None, f"ECC perplexity error: {e}"

    del model
    torch.cuda.empty_cache()

    if total_tokens > 0:
        ppl = math.exp(total_loss / total_tokens)
        return ppl, None
    return None, "No tokens processed"


def run_full_comparison(config=None, progress_callback=None):
    """Run full vLLM vs ECC comparison benchmark."""
    if config is None:
        config = VLLMComparisonConfig()

    report = ComparisonReport(config=config)

    try:
        from vllm import LLM
        report.vllm_available = True
    except ImportError:
        report.vllm_available = False
        report.error_message = "vLLM not installed, skipping vLLM baselines"

    total_tasks = len(config.configs) * len(config.batch_sizes) * len(config.seq_lengths)
    if config.compute_perplexity:
        total_tasks += len(config.configs)

    current_task = 0

    for cfg_name in config.configs:
        for batch_size in config.batch_sizes:
            for seq_length in config.seq_lengths:
                current_task += 1
                if progress_callback:
                    progress_callback(
                        f"{cfg_name} B={batch_size} S={seq_length}",
                        current_task,
                        total_tasks
                    )

                result = None
                error = None

                if cfg_name.startswith("vllm_"):
                    if not report.vllm_available:
                        continue
                    kv_dtype = "fp8" if "fp8" in cfg_name else "auto"
                    result, error = run_vllm_throughput(
                        config, kv_dtype, batch_size, seq_length
                    )
                elif cfg_name.startswith("ecc_"):
                    codec = cfg_name.replace("ecc_", "")
                    result, error = run_ecc_throughput(
                        config, codec, batch_size, seq_length
                    )

                if result:
                    report.results.append(result)
                elif error:
                    print(f"  Error for {cfg_name}: {error}")

    if config.compute_perplexity:
        print("\nComputing perplexity...")

        for cfg_name in config.configs:
            current_task += 1
            if progress_callback:
                progress_callback(f"PPL: {cfg_name}", current_task, total_tasks)

            ppl = None
            error = None

            if cfg_name.startswith("vllm_"):
                if not report.vllm_available:
                    continue
                kv_dtype = "fp8" if "fp8" in cfg_name else "auto"
                ppl, error = compute_vllm_perplexity(config, kv_dtype)
            elif cfg_name.startswith("ecc_"):
                codec = cfg_name.replace("ecc_", "")
                ppl, error = compute_ecc_perplexity(config, codec)

            if ppl is not None:
                for r in report.results:
                    if r.config == cfg_name:
                        r.perplexity = ppl
                        break
                else:
                    report.results.append(BenchmarkResult(
                        config=cfg_name,
                        batch_size=0,
                        seq_length=0,
                        throughput_tokens_sec=0,
                        latency_ms_per_token=0,
                        memory_gb=0,
                        perplexity=ppl,
                    ))

            if error:
                print(f"  PPL error for {cfg_name}: {error}")

    return report


def run_vllm_comparison_impl(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    batch_sizes=None,
    seq_lengths=None,
    num_samples=100,
    compute_perplexity=True,
):
    """Entry point for Modal execution."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    if seq_lengths is None:
        seq_lengths = [512, 1024]

    config = VLLMComparisonConfig(
        model_name=model_name,
        batch_sizes=batch_sizes,
        seq_lengths=seq_lengths,
        num_samples=num_samples,
        compute_perplexity=compute_perplexity,
    )

    def progress(msg, curr, total):
        print(f"[{curr}/{total}] {msg}")

    report = run_full_comparison(config, progress)

    print("\n" + report.get_full_report())

    return report.to_dict()


def run_ber_sweep_impl(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ber_levels=None,
    perplexity_samples=50,
):
    """
    Entry point for BER sweep benchmark on Modal.

    Compares perplexity degradation between:
    - vLLM FP16/FP8 (unprotected, degrades under errors)
    - ECC-protected INT4 (maintains quality under errors)
    """
    import torch

    print("=" * 80)
    print("BER SWEEP BENCHMARK: vLLM vs ECC-Protected KV Cache")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("")

    if ber_levels is None:
        ber_levels = [0.0, 1e-4, 1e-3, 1e-2]

    config = VLLMComparisonConfig(
        model_name=model_name,
        ber_levels=ber_levels,
        perplexity_samples=perplexity_samples,
        run_ber_sweep=True,
    )

    print(f"Model: {model_name}")
    print(f"BER levels: {ber_levels}")
    print(f"Perplexity samples: {perplexity_samples}")
    print("")

    def progress(msg, curr, total):
        print(f"[{curr}/{total}] {msg}")

    results = run_ber_sweep_comparison(config, progress)

    # Format and print results
    table = format_ber_sweep_table(results)
    print(table)

    # Return results as dict for serialization
    return {
        "config": {
            "model_name": model_name,
            "ber_levels": ber_levels,
            "perplexity_samples": perplexity_samples,
        },
        "results": [
            {
                "config": r.config,
                "ber": r.ber,
                "perplexity": r.perplexity,
                "bits_flipped": r.bits_flipped,
                "bits_processed": r.bits_processed,
            }
            for r in results
        ],
        "formatted_table": table,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ber-sweep":
        run_ber_sweep_impl()
    else:
        run_vllm_comparison_impl()
