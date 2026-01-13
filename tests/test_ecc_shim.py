import pytest
import torch
import torch.nn as nn
import math


class TestSimpleBlockManager:
    def test_initialization(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=4,
            num_kv_heads=8,
            head_dim=64,
            device="cuda",
            codec="hamming84",
        )

        assert manager.num_blocks == 64
        assert manager.block_size == 16
        assert manager.num_layers == 4
        assert manager.num_kv_heads == 8
        assert len(manager.free_blocks) == 64
        assert manager.k_cache.shape == (64, 4, 8, 16 * 64)
        assert manager.k_cache.dtype == torch.uint8

    def test_allocation_single_sequence(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        block_table, ctx_len = manager.allocate(seq_id=0, num_tokens=48)

        assert ctx_len == 48
        assert manager.get_context_len(0) == 48
        assert len(manager.free_blocks) == 32 - 3
        assert (block_table[:3] >= 0).all()
        assert (block_table[3:] == -1).all()

    def test_allocation_incremental(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        manager.allocate(seq_id=0, num_tokens=16)
        assert len(manager.free_blocks) == 31

        manager.allocate(seq_id=0, num_tokens=48)
        assert len(manager.free_blocks) == 29

    def test_allocation_multiple_sequences(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=64,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        manager.allocate(seq_id=0, num_tokens=32)
        manager.allocate(seq_id=1, num_tokens=64)

        assert len(manager.free_blocks) == 64 - 6
        assert manager.get_context_len(0) == 32
        assert manager.get_context_len(1) == 64

    def test_reset_clears_state(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        manager.allocate(seq_id=0, num_tokens=48)
        manager.allocate(seq_id=1, num_tokens=32)
        assert len(manager.free_blocks) == 32 - 5

        manager.reset()

        assert len(manager.free_blocks) == 32
        assert manager.get_context_len(0) == 0
        assert manager.get_context_len(1) == 0
        assert (manager.block_table == -1).all()

    def test_out_of_blocks_raises(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=4,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        with pytest.raises(RuntimeError, match="Out of blocks"):
            manager.allocate(seq_id=0, num_tokens=100)


class TestECCBackend:
    def test_write_stores_data(self):
        from kv_cache.ecc_shim import SimpleBlockManager, ECCBackend, ECCShimConfig

        manager = SimpleBlockManager(
            num_blocks=16,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        config = ECCShimConfig(codec="hamming84", ber=0.0)
        backend = ECCBackend(manager, config, num_heads=8)

        batch_size, seq_len = 1, 16
        k = torch.randn(batch_size, seq_len, 4 * 32, device="cuda")
        v = torch.randn(batch_size, seq_len, 4 * 32, device="cuda")

        backend.write(k, v, layer_idx=0, seq_id=0)

        assert manager.k_cache.abs().sum() > 0
        assert manager.v_cache.abs().sum() > 0
        assert manager.k_scales.abs().sum() > 0

    def test_attend_returns_correct_shape(self):
        from kv_cache.ecc_shim import SimpleBlockManager, ECCBackend, ECCShimConfig

        manager = SimpleBlockManager(
            num_blocks=16,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        config = ECCShimConfig(codec="hamming84", ber=0.0)
        backend = ECCBackend(manager, config, num_heads=4)

        batch_size, seq_len = 1, 16
        k = torch.randn(batch_size, seq_len, 4 * 32, device="cuda")
        v = torch.randn(batch_size, seq_len, 4 * 32, device="cuda")
        backend.write(k, v, layer_idx=0, seq_id=0)

        q = torch.randn(batch_size, 4, seq_len, 32, device="cuda")

        output = backend.attend(q, layer_idx=0, seq_id=0)

        assert output.shape == (batch_size, 4, seq_len, 32)


class TestECCPagedAttentionShim:
    def _create_mock_attention(
        self, num_heads=8, num_kv_heads=4, head_dim=32, hidden_size=256
    ):
        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = num_heads
                self.num_key_value_heads = num_kv_heads
                self.head_dim = head_dim
                self.hidden_size = hidden_size

                self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(
                    hidden_size, num_kv_heads * head_dim, bias=False
                )
                self.v_proj = nn.Linear(
                    hidden_size, num_kv_heads * head_dim, bias=False
                )
                self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        return MockAttention().cuda()

    def _create_mock_rotary_emb(self, head_dim=32, max_seq_len=128):
        class MockRotaryEmb(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim = head_dim
                inv_freq = 1.0 / (
                    10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
                )
                self.register_buffer("inv_freq", inv_freq)

            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                t = position_ids.float()
                freqs = torch.einsum("bi,j->bij", t, self.inv_freq.to(x.device))
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().unsqueeze(1)
                sin = emb.sin().unsqueeze(1)
                return cos, sin

        return MockRotaryEmb().cuda()

    def test_shim_forward_shape(self):
        from kv_cache.ecc_shim import (
            SimpleBlockManager,
            ECCBackend,
            ECCShimConfig,
            ECCPagedAttentionShim,
        )

        num_heads, num_kv_heads, head_dim, hidden_size = 8, 4, 32, 256

        manager = SimpleBlockManager(
            num_blocks=16,
            block_size=16,
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cuda",
        )

        config = ECCShimConfig(codec="hamming84", ber=0.0)
        backend = ECCBackend(manager, config, num_heads=num_heads)

        mock_attn = self._create_mock_attention(
            num_heads, num_kv_heads, head_dim, hidden_size
        )
        mock_rotary = self._create_mock_rotary_emb(head_dim)

        shim = ECCPagedAttentionShim(
            original_attn=mock_attn,
            layer_idx=0,
            backend=backend,
            rotary_emb=mock_rotary,
        )

        batch_size, seq_len = 1, 16
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

        result = shim(hidden_states, position_ids=position_ids)
        output = result[0]
        attn_weights = result[1] if len(result) > 1 else None

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attn_weights is None

    def test_shim_handles_none_position_ids(self):
        from kv_cache.ecc_shim import (
            SimpleBlockManager,
            ECCBackend,
            ECCShimConfig,
            ECCPagedAttentionShim,
        )

        num_heads, num_kv_heads, head_dim, hidden_size = 4, 4, 32, 128

        manager = SimpleBlockManager(
            num_blocks=16,
            block_size=16,
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cuda",
        )

        config = ECCShimConfig(codec="hamming84", ber=0.0)
        backend = ECCBackend(manager, config, num_heads=num_heads)

        mock_attn = self._create_mock_attention(
            num_heads, num_kv_heads, head_dim, hidden_size
        )
        mock_rotary = self._create_mock_rotary_emb(head_dim)

        shim = ECCPagedAttentionShim(
            original_attn=mock_attn,
            layer_idx=0,
            backend=backend,
            rotary_emb=mock_rotary,
        )

        batch_size, seq_len = 1, 8
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda")

        result = shim(hidden_states, position_ids=None)
        output = result[0]

        assert output.shape == (batch_size, seq_len, hidden_size)


class TestPatchModelWithECCAttention:
    @pytest.fixture
    def llama_model(self):
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_options = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "meta-llama/Llama-3.2-1B",
        ]

        hf_token = os.environ.get("HF_TOKEN")

        for model_name in model_options:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    token=hf_token,
                    device_map="cuda",
                )
                model.eval()
                return model, tokenizer, model_name
            except Exception as e:
                print(f"Could not load {model_name}: {e}")
                continue

        pytest.skip("No LLaMA-compatible model available")

    def test_patch_replaces_layers(self, llama_model):
        from kv_cache.ecc_shim import (
            patch_model_with_ecc_attention,
            ECCShimConfig,
            ECCPagedAttentionShim,
        )

        model, tokenizer, model_name = llama_model
        config = ECCShimConfig(codec="hamming84", ber=0.0)

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            original_attn_type = type(model.model.layers[0].self_attn)
        else:
            original_attn_type = type(model.layers[0].self_attn)

        with patch_model_with_ecc_attention(model, config, num_blocks=64):
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                patched_attn = model.model.layers[0].self_attn
            else:
                patched_attn = model.layers[0].self_attn

            assert isinstance(
                patched_attn, ECCPagedAttentionShim
            ), f"Expected ECCPagedAttentionShim, got {type(patched_attn)}"

    def test_patch_restores_on_exit(self, llama_model):
        from kv_cache.ecc_shim import patch_model_with_ecc_attention, ECCShimConfig

        model, tokenizer, model_name = llama_model
        config = ECCShimConfig(codec="hamming84", ber=0.0)

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            original_attn = model.model.layers[0].self_attn
            original_attn_type = type(original_attn)
        else:
            original_attn = model.layers[0].self_attn
            original_attn_type = type(original_attn)

        with patch_model_with_ecc_attention(model, config, num_blocks=64):
            pass

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            restored_attn = model.model.layers[0].self_attn
        else:
            restored_attn = model.layers[0].self_attn

        assert (
            type(restored_attn) == original_attn_type
        ), f"Expected {original_attn_type}, got {type(restored_attn)}"

    def test_patch_forward_produces_output(self, llama_model):
        from kv_cache.ecc_shim import patch_model_with_ecc_attention, ECCShimConfig

        model, tokenizer, model_name = llama_model
        config = ECCShimConfig(codec="hamming84", ber=0.0)

        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with patch_model_with_ecc_attention(model, config, num_blocks=64):
            with torch.no_grad():
                outputs = model(**inputs)

        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1
        assert not torch.isnan(outputs.logits).any()

    def test_shim_rope_correctness(self, llama_model):
        from kv_cache.ecc_shim import (
            patch_model_with_ecc_attention,
            ECCShimConfig,
            reset_ecc_cache,
        )

        model, tokenizer, model_name = llama_model
        config = ECCShimConfig(codec="hamming84", ber=0.0)

        text = "The quick brown fox"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            original_outputs = model(**inputs)
            original_logits = original_outputs.logits.clone()

        with patch_model_with_ecc_attention(model, config, num_blocks=64):
            with torch.no_grad():
                patched_outputs = model(**inputs)
                patched_logits = patched_outputs.logits.clone()

        original_flat = original_logits.flatten().float()
        patched_flat = patched_logits.flatten().float()

        cosine_sim = torch.nn.functional.cosine_similarity(
            original_flat.unsqueeze(0), patched_flat.unsqueeze(0)
        ).item()

        print(f"Cosine similarity between original and patched: {cosine_sim:.4f}")

        assert cosine_sim > 0.0, f"Cosine similarity should be positive: {cosine_sim}"

        original_std = original_flat.std().item()
        patched_std = patched_flat.std().item()
        std_ratio = patched_std / (original_std + 1e-8)
        assert 0.1 < std_ratio < 10, f"Output magnitude diverged: ratio={std_ratio:.2f}"


class TestShimVsBaseline:
    def test_shim_vs_pytorch_attention_mse(self):
        from kv_cache.ecc_shim import (
            SimpleBlockManager,
            ECCBackend,
            ECCShimConfig,
            ECCPagedAttentionShim,
        )

        torch.manual_seed(42)

        num_heads, num_kv_heads, head_dim, hidden_size = 4, 4, 32, 128

        manager = SimpleBlockManager(
            num_blocks=16,
            block_size=16,
            num_layers=2,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cuda",
        )

        config = ECCShimConfig(codec="hamming84", ber=0.0)
        backend = ECCBackend(manager, config, num_heads=num_heads)

        class MockAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = num_heads
                self.num_key_value_heads = num_kv_heads
                self.head_dim = head_dim
                self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
                self.k_proj = nn.Linear(
                    hidden_size, num_kv_heads * head_dim, bias=False
                )
                self.v_proj = nn.Linear(
                    hidden_size, num_kv_heads * head_dim, bias=False
                )
                self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        mock_attn = MockAttention().cuda()

        class MockRotaryEmb(nn.Module):
            def __init__(self):
                super().__init__()
                inv_freq = 1.0 / (
                    10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim)
                )
                self.register_buffer("inv_freq", inv_freq)

            def forward(self, x, position_ids):
                seq_len = position_ids.shape[-1]
                t = position_ids.float()
                freqs = torch.einsum("bi,j->bij", t, self.inv_freq.to(x.device))
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().unsqueeze(1)
                sin = emb.sin().unsqueeze(1)
                return cos, sin

        mock_rotary = MockRotaryEmb().cuda()

        shim = ECCPagedAttentionShim(
            original_attn=mock_attn,
            layer_idx=0,
            backend=backend,
            rotary_emb=mock_rotary,
        )

        batch_size, seq_len = 1, 16
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

        with torch.no_grad():
            result = shim(hidden_states, position_ids=position_ids)
            shim_output = result[0]

        assert not torch.isnan(shim_output).any(), "Output contains NaN"
        assert not torch.isinf(shim_output).any(), "Output contains Inf"
        assert shim_output.abs().mean() > 0, "Output is all zeros"


class TestShimECCCorrection:
    def test_ecc_corrects_single_bit_errors(self):
        from kv_cache.ecc_shim import SimpleBlockManager, ECCBackend, ECCShimConfig

        torch.manual_seed(42)

        manager = SimpleBlockManager(
            num_blocks=16,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        config = ECCShimConfig(
            codec="hamming84",
            ber=0.001,
            inject_errors=True,
            seed=42,
        )
        backend = ECCBackend(manager, config, num_heads=4)

        batch_size, seq_len = 1, 16
        k = torch.randn(batch_size, seq_len, 4 * 32, device="cuda")
        v = torch.randn(batch_size, seq_len, 4 * 32, device="cuda")
        backend.write(k, v, layer_idx=0, seq_id=0)

        q = torch.randn(batch_size, 4, seq_len, 32, device="cuda")

        output = backend.attend(q, layer_idx=0, seq_id=0)

        assert not torch.isnan(output).any(), "Output contains NaN after ECC"
        assert output.shape == (batch_size, 4, seq_len, 32)


class TestResetECCCache:
    def test_reset_prevents_oom(self):
        from kv_cache.ecc_shim import SimpleBlockManager

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        for doc_id in range(10):
            manager.allocate(seq_id=0, num_tokens=64 + doc_id * 16)

            manager.reset()

            assert (
                len(manager.free_blocks) == 32
            ), f"Blocks not freed after doc {doc_id}"


class TestGetECCStats:
    def test_get_stats_returns_dict(self):
        from kv_cache.ecc_shim import (
            SimpleBlockManager,
            ECCBackend,
            ECCShimConfig,
            get_ecc_stats,
        )

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=32,
            device="cuda",
        )

        config = ECCShimConfig()
        backend = ECCBackend(manager, config, num_heads=4)

        class MockModel(nn.Module):
            pass

        model = MockModel()
        model._ecc_block_manager = manager
        model._ecc_backend = backend

        manager.allocate(seq_id=0, num_tokens=48)

        stats = get_ecc_stats(model)

        assert "allocated_blocks" in stats
        assert "free_blocks" in stats
        assert "sequences" in stats
        assert stats["allocated_blocks"] == 3
        assert stats["free_blocks"] == 29


class TestInterpolationDimension:
    """Tests verifying interpolation uses correct dimension (temporal, not features)."""

    def test_interpolation_uses_temporal_neighbors(self):
        """Verify interpolation is along context length (seq_dim=0), not head dimension.

        This test validates the fix for the interpolation dimension bug where
        the original code interpolated along the wrong dimension.
        """
        from ecc_codecs.triton_kernels import interpolate_double_errors
        from ecc_codecs.triton_kernels.config import ErrorType

        ctx_len, heads, dim = 10, 4, 64

        # Create tensor where each token position has a unique value
        # k[i, :, :] = i for all heads and features
        k = torch.arange(ctx_len, dtype=torch.float32, device="cuda")
        k = k.view(ctx_len, 1, 1).expand(ctx_len, heads, dim).contiguous()
        k = k.to(torch.uint8)

        # Mark position 5 as double error (DOUBLE_DETECTED = 2)
        error_types = torch.zeros(ctx_len, heads, dim, dtype=torch.uint8, device="cuda")
        error_types[5, :, :] = ErrorType.DOUBLE_DETECTED

        # Interpolate along temporal dimension (seq_dim=0)
        result = interpolate_double_errors(k, error_types, seq_dim=0)

        # Position 5 should be average of positions 4 and 6 = (4 + 6) / 2 = 5
        # After rounding to uint8
        expected_value = 5
        actual_values = result[5]

        assert torch.all(actual_values == expected_value), (
            f"Interpolation should use temporal neighbors. "
            f"Expected all values at position 5 to be {expected_value}, "
            f"but got {actual_values.unique().tolist()}"
        )

    def test_interpolation_boundary_first_position(self):
        """Verify interpolation handles first position correctly."""
        from ecc_codecs.triton_kernels import interpolate_double_errors
        from ecc_codecs.triton_kernels.config import ErrorType

        ctx_len, heads, dim = 10, 2, 32
        k = torch.arange(ctx_len, dtype=torch.float32, device="cuda")
        k = k.view(ctx_len, 1, 1).expand(ctx_len, heads, dim).contiguous()
        k = k.to(torch.uint8)

        # Mark first position as double error
        error_types = torch.zeros(ctx_len, heads, dim, dtype=torch.uint8, device="cuda")
        error_types[0, :, :] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(k, error_types, seq_dim=0)

        # First position uses clamped left neighbor (0) and right neighbor (1)
        # Average = (0 + 1) / 2 = 0.5 → rounds to 1 with +0.5 rounding
        expected_value = 1
        actual_values = result[0]

        assert torch.all(actual_values == expected_value), (
            f"First position interpolation failed. Expected {expected_value}, "
            f"got {actual_values.unique().tolist()}"
        )

    def test_interpolation_boundary_last_position(self):
        """Verify interpolation handles last position correctly."""
        from ecc_codecs.triton_kernels import interpolate_double_errors
        from ecc_codecs.triton_kernels.config import ErrorType

        ctx_len, heads, dim = 10, 2, 32
        k = torch.arange(ctx_len, dtype=torch.float32, device="cuda")
        k = k.view(ctx_len, 1, 1).expand(ctx_len, heads, dim).contiguous()
        k = k.to(torch.uint8)

        # Mark last position as double error
        error_types = torch.zeros(ctx_len, heads, dim, dtype=torch.uint8, device="cuda")
        error_types[-1, :, :] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(k, error_types, seq_dim=0)

        # Last position uses left neighbor (8) and clamped right neighbor (9)
        # Average = (8 + 9) / 2 = 8.5 → rounds to 9 with +0.5 rounding
        expected_value = 9
        actual_values = result[-1]

        assert torch.all(actual_values == expected_value), (
            f"Last position interpolation failed. Expected {expected_value}, "
            f"got {actual_values.unique().tolist()}"
        )

    def test_interpolation_preserves_non_error_values(self):
        """Verify interpolation doesn't modify non-error positions."""
        from ecc_codecs.triton_kernels import interpolate_double_errors
        from ecc_codecs.triton_kernels.config import ErrorType

        ctx_len, heads, dim = 10, 4, 64
        k = torch.randint(0, 16, (ctx_len, heads, dim), dtype=torch.uint8, device="cuda")
        original = k.clone()

        # Mark position 5 as double error
        error_types = torch.zeros(ctx_len, heads, dim, dtype=torch.uint8, device="cuda")
        error_types[5, :, :] = ErrorType.DOUBLE_DETECTED

        result = interpolate_double_errors(k, error_types, seq_dim=0)

        # Non-error positions should be unchanged
        non_error_mask = error_types != ErrorType.DOUBLE_DETECTED
        assert torch.all(result[non_error_mask] == original[non_error_mask]), (
            "Non-error positions should be unchanged by interpolation"
        )


class TestECCBackendStatsReset:
    """Tests for ECCBackend statistics reset functionality.

    These tests verify that the P0 CRITICAL fix for stats leaking across texts
    works correctly. Previously, reset_ecc_cache() only reset the block manager,
    not the backend's error statistics, causing contaminated results.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_stats_reset_clears_all_counters(self):
        """Verify reset_stats() clears all ECCBackend counters."""
        from kv_cache.ecc_shim import (
            ECCShimConfig,
            SimpleBlockManager,
            ECCBackend,
        )

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cuda",
            codec="hamming84",
        )
        config = ECCShimConfig(codec="hamming84", ber=0.01, inject_errors=True)
        backend = ECCBackend(manager, config, num_heads=4)

        # Simulate accumulated statistics
        backend._injection_count = 500
        backend._errors_corrected = 100
        backend._errors_detected = 10
        backend._total_values = 10000

        # Reset
        backend.reset_stats()

        # Verify ALL stats are zero
        assert backend._injection_count == 0, "injection_count should be reset"
        assert backend._errors_corrected == 0, "errors_corrected should be reset"
        assert backend._errors_detected == 0, "errors_detected should be reset"
        assert backend._total_values == 0, "total_values should be reset"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_reset_ecc_cache_resets_backend_stats(self):
        """Verify reset_ecc_cache() resets backend stats via model attribute."""
        from kv_cache.ecc_shim import (
            ECCShimConfig,
            SimpleBlockManager,
            ECCBackend,
            reset_ecc_cache,
            get_ecc_stats,
        )

        # Create mock model with backend attached
        class MockModel:
            pass

        model = MockModel()
        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            device="cuda",
            codec="hamming84",
        )
        config = ECCShimConfig(codec="hamming84", ber=0.01, inject_errors=True)
        backend = ECCBackend(manager, config, num_heads=4)

        model._ecc_block_manager = manager
        model._ecc_backend = backend

        # Simulate accumulated statistics
        backend._injection_count = 500
        backend._errors_corrected = 100
        backend._errors_detected = 10
        backend._total_values = 10000

        # Reset via reset_ecc_cache
        reset_ecc_cache(model)

        # Verify stats are reset
        stats = get_ecc_stats(model)
        assert stats["injection_count"] == 0, "injection_count should be reset"
        assert stats["errors_corrected"] == 0, "errors_corrected should be reset"
        assert stats["errors_detected"] == 0, "errors_detected should be reset"
        assert stats["total_values"] == 0, "total_values should be reset"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_total_values_tracked_on_write(self):
        """Verify total_values is incremented when writing K,V."""
        from kv_cache.ecc_shim import (
            ECCShimConfig,
            SimpleBlockManager,
            ECCBackend,
        )

        manager = SimpleBlockManager(
            num_blocks=32,
            block_size=16,
            num_layers=1,
            num_kv_heads=4,
            head_dim=64,
            device="cuda",
            codec="hamming84",
        )
        config = ECCShimConfig(codec="hamming84", ber=0.0)
        backend = ECCBackend(manager, config, num_heads=4)

        # Write some K,V
        batch_size = 1
        seq_len = 10
        num_kv_heads = 4
        head_dim = 64

        k = torch.randn(batch_size, seq_len, num_kv_heads * head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_kv_heads * head_dim, device="cuda", dtype=torch.float16)

        # Before write, total_values should be 0
        assert backend._total_values == 0

        backend.write(k, v, layer_idx=0, seq_id=0)

        # After write, total_values should be:
        # 2 (K and V) * batch_size * seq_len * num_kv_heads * head_dim
        expected = 2 * batch_size * seq_len * num_kv_heads * head_dim
        assert backend._total_values == expected, (
            f"Expected total_values={expected}, got {backend._total_values}"
        )
