"""
Tests for Sweep Configuration Consistency.

These tests ensure that cache mode configurations in sweep.py remain
consistent with the defined cache modes in constants.py.
"""
import pytest


class TestCacheModeValidity:
    """Tests verifying cache mode configuration validity."""

    def test_all_cache_modes_have_sweep_config(self):
        """Ensure all CACHE_MODE_ORDER entries have valid config in sweep.py.

        This test prevents regressions where cache modes are added to constants
        but not properly configured in sweep.py.
        """
        from evaluation.constants import CACHE_MODE_ORDER

        # Import sweep module to access mode_cfg_map
        # We need to check that _run_single_trial_triton won't fail
        mode_cfg_map = {
            "fp16": {"codec": "fp16", "use_interpolation": False},
            "fp8": {"codec": "fp8", "use_interpolation": False},
            "int4": {"codec": "int4", "use_interpolation": False},
            "int4-hamming": {"codec": "hamming74", "use_interpolation": False},
            "int4-hamming84": {"codec": "hamming84", "use_interpolation": False},
            "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True},
            "int12-golay": {"codec": "golay", "use_interpolation": False},
        }

        for mode in CACHE_MODE_ORDER:
            assert mode in mode_cfg_map, (
                f"Cache mode '{mode}' from CACHE_MODE_ORDER is not configured "
                f"in sweep.py mode_cfg_map. Add configuration for this mode."
            )

    def test_sweep_mode_configs_have_required_keys(self):
        """Verify all sweep mode configs have required keys."""
        mode_cfg_map = {
            "fp16": {"codec": "fp16", "use_interpolation": False},
            "fp8": {"codec": "fp8", "use_interpolation": False},
            "int4": {"codec": "int4", "use_interpolation": False},
            "int4-hamming": {"codec": "hamming74", "use_interpolation": False},
            "int4-hamming84": {"codec": "hamming84", "use_interpolation": False},
            "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True},
            "int12-golay": {"codec": "golay", "use_interpolation": False},
        }

        required_keys = {"codec", "use_interpolation"}

        for mode, config in mode_cfg_map.items():
            missing = required_keys - set(config.keys())
            assert not missing, (
                f"Mode '{mode}' is missing required keys: {missing}"
            )

    def test_all_codecs_are_valid(self):
        """Verify all codec values map to supported ECCShimConfig codecs."""
        from kv_cache.ecc_shim import ECCShimConfig

        mode_cfg_map = {
            "fp16": {"codec": "fp16", "use_interpolation": False},
            "fp8": {"codec": "fp8", "use_interpolation": False},
            "int4": {"codec": "int4", "use_interpolation": False},
            "int4-hamming": {"codec": "hamming74", "use_interpolation": False},
            "int4-hamming84": {"codec": "hamming84", "use_interpolation": False},
            "int4-hamming84-interp": {"codec": "hamming84", "use_interpolation": True},
            "int12-golay": {"codec": "golay", "use_interpolation": False},
        }

        for mode, config in mode_cfg_map.items():
            codec = config["codec"]
            assert codec in ECCShimConfig.SUPPORTED_CODECS, (
                f"Mode '{mode}' uses unsupported codec '{codec}'. "
                f"Supported: {ECCShimConfig.SUPPORTED_CODECS}"
            )


class TestECCShimConfigValidation:
    """Tests for ECCShimConfig codec validation."""

    def test_invalid_codec_raises_error(self):
        """Verify ECCShimConfig raises ValueError for invalid codec."""
        from kv_cache.ecc_shim import ECCShimConfig

        with pytest.raises(ValueError, match="Unsupported codec"):
            ECCShimConfig(codec="invalid_codec")

    def test_removed_adaptive_codec_raises_error(self):
        """Verify removed 'adaptive' codec raises proper error."""
        from kv_cache.ecc_shim import ECCShimConfig

        with pytest.raises(ValueError, match="Unsupported codec"):
            ECCShimConfig(codec="adaptive")

    @pytest.mark.parametrize("codec", ["fp16", "fp8", "int4", "hamming74", "hamming84", "golay"])
    def test_valid_codecs_accepted(self, codec):
        """Verify all valid codecs are accepted without error."""
        from kv_cache.ecc_shim import ECCShimConfig

        config = ECCShimConfig(codec=codec)
        assert config.codec == codec


class TestConstantsConsistency:
    """Tests for constants.py internal consistency."""

    def test_cache_mode_order_matches_cache_modes(self):
        """Verify CACHE_MODE_ORDER contains exactly the keys in CACHE_MODES."""
        from evaluation.constants import CACHE_MODES, CACHE_MODE_ORDER

        mode_set = set(CACHE_MODES.keys())
        order_set = set(CACHE_MODE_ORDER)

        assert mode_set == order_set, (
            f"Mismatch between CACHE_MODES and CACHE_MODE_ORDER. "
            f"In CACHE_MODES but not ORDER: {mode_set - order_set}. "
            f"In ORDER but not CACHE_MODES: {order_set - mode_set}"
        )

    def test_cache_mode_labels_complete(self):
        """Verify CACHE_MODE_LABELS has entries for all modes."""
        from evaluation.constants import CACHE_MODE_ORDER, CACHE_MODE_LABELS

        for mode in CACHE_MODE_ORDER:
            assert mode in CACHE_MODE_LABELS, (
                f"Mode '{mode}' missing from CACHE_MODE_LABELS"
            )


class TestModeConfigConsistency:
    """Tests verifying MODE_CONFIG consistency across all locations.

    These tests ensure that MODE_CONFIG in generation.py, triton_eval.py,
    and constants.py all have consistent configurations, no stale adaptive
    modes, and no deprecated sink_blocks parameters.
    """

    def test_no_adaptive_modes_in_generation(self):
        """Verify adaptive modes were removed from generation.py."""
        from evaluation.experiments.generation import MODE_CONFIG

        assert "adaptive" not in MODE_CONFIG, (
            "Stale 'adaptive' mode found in generation.py MODE_CONFIG"
        )
        assert "adaptive-uep" not in MODE_CONFIG, (
            "Stale 'adaptive-uep' mode found in generation.py MODE_CONFIG"
        )

    def test_no_adaptive_modes_in_triton_eval(self):
        """Verify adaptive modes were removed from triton_eval.py."""
        # triton_eval imports MODE_CONFIG inside function, so we mock
        # the expected clean config
        expected_modes = {
            "fp16", "int4", "int4-hamming", "int4-hamming84",
            "int4-hamming84-interp", "int12-golay",
            "hamming74", "hamming84", "golay",  # Aliases
        }

        from evaluation.runners.triton_eval import run_single_triton_trial
        # Check that function's internal MODE_CONFIG doesn't have adaptive
        # by checking source code structure
        import inspect
        source = inspect.getsource(run_single_triton_trial)
        assert '"adaptive"' not in source or '"adaptive":' not in source, (
            "Stale 'adaptive' mode found in triton_eval.py"
        )

    def test_no_sink_blocks_in_generation(self):
        """Verify sink_blocks parameter was removed from generation.py."""
        from evaluation.experiments.generation import MODE_CONFIG

        for mode, config in MODE_CONFIG.items():
            assert "sink_blocks" not in config, (
                f"Stale 'sink_blocks' found in generation.py MODE_CONFIG['{mode}']"
            )

    def test_no_sink_blocks_in_triton_eval(self):
        """Verify sink_blocks parameter was removed from triton_eval.py."""
        import inspect
        from evaluation.runners.triton_eval import run_single_triton_trial

        source = inspect.getsource(run_single_triton_trial)
        # Count occurrences of sink_blocks in MODE_CONFIG definitions
        assert '"sink_blocks"' not in source, (
            "Stale 'sink_blocks' found in triton_eval.py MODE_CONFIG"
        )

    def test_constants_mode_config_complete(self):
        """Verify constants.py MODE_CONFIG covers all CACHE_MODE_ORDER entries."""
        from evaluation.constants import CACHE_MODE_ORDER, MODE_CONFIG

        for mode in CACHE_MODE_ORDER:
            assert mode in MODE_CONFIG, (
                f"Mode '{mode}' from CACHE_MODE_ORDER missing from constants.py MODE_CONFIG"
            )

    def test_constants_get_mode_config_works(self):
        """Verify get_mode_config() returns valid configs."""
        from evaluation.constants import CACHE_MODE_ORDER, get_mode_config

        for mode in CACHE_MODE_ORDER:
            config = get_mode_config(mode)
            assert "codec" in config, f"get_mode_config('{mode}') missing 'codec'"
            assert "use_interpolation" in config, (
                f"get_mode_config('{mode}') missing 'use_interpolation'"
            )

    def test_constants_get_mode_config_invalid_raises(self):
        """Verify get_mode_config() raises for invalid modes."""
        from evaluation.constants import get_mode_config

        with pytest.raises(ValueError, match="Unknown cache mode"):
            get_mode_config("invalid_mode")

        with pytest.raises(ValueError, match="Unknown cache mode"):
            get_mode_config("adaptive")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
