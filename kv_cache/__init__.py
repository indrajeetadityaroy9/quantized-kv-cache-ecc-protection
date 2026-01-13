from .memory_layout import (
    ECCCacheConfig,
    allocate_ecc_kv_cache,
    get_physical_block,
    create_block_table,
)

from .attention_ecc import (
    paged_attention_ecc,
    reference_attention_ecc,
)

from .ecc_shim import (
    SimpleBlockManager,
    ECCShimConfig,
    ECCDummyCache,
    ECCBackend,
    ECCPagedAttentionShim,
    patch_model_with_ecc_attention,
    reset_ecc_cache,
    get_ecc_stats,
    _get_attention_params,
    _find_rotary_embedding,
    compute_injection_seed,
)

from .unprotected_shim import (
    UnprotectedShimConfig,
    UnprotectedDummyCache,
    UnprotectedBackend,
    UnprotectedPagedAttentionShim,
    patch_model_with_unprotected_attention,
    reset_unprotected_cache,
    get_unprotected_stats,
)

__all__ = [
    "ECCCacheConfig",
    "allocate_ecc_kv_cache",
    "get_physical_block",
    "create_block_table",
    "paged_attention_ecc",
    "reference_attention_ecc",
    "SimpleBlockManager",
    "ECCShimConfig",
    "ECCDummyCache",
    "ECCBackend",
    "ECCPagedAttentionShim",
    "patch_model_with_ecc_attention",
    "reset_ecc_cache",
    "get_ecc_stats",
    "_get_attention_params",
    "_find_rotary_embedding",
    "compute_injection_seed",
    "UnprotectedShimConfig",
    "UnprotectedDummyCache",
    "UnprotectedBackend",
    "UnprotectedPagedAttentionShim",
    "patch_model_with_unprotected_attention",
    "reset_unprotected_cache",
    "get_unprotected_stats",
]
