from .modal import (
    setup_modal_environment,
    get_gpu_info,
    run_tests_impl,
)

# Lazy imports for vLLM runner - import on demand to allow sys.path modification
def get_vllm_runner():
    """Get VLLMEvaluationRunner class (lazy import)."""
    from .vllm_runner import VLLMEvaluationRunner
    return VLLMEvaluationRunner


def get_cache_mode_mapping():
    """Get cache mode to vLLM dtype mapping."""
    from .vllm_runner import CACHE_MODE_TO_VLLM_DTYPE
    return CACHE_MODE_TO_VLLM_DTYPE


__all__ = [
    "setup_modal_environment",
    "get_gpu_info",
    "run_tests_impl",
    "get_vllm_runner",
    "get_cache_mode_mapping",
]
