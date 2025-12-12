"""
vLLM PagedAttention ECC Integration Kernels.

This package provides GPU-native integration of ECC (Hamming/Golay) codecs
with vLLM's PagedAttention KV cache management.

Components:
- memory_layout: Block table and cache allocation utilities
- paged_cache_ecc: ECC-integrated cache write kernel
- attention_ecc: ECC-integrated attention read kernel
- adaptive_uep: Position-based codec routing (sink vs context)
- benchmark_harness: Performance measurement utilities
"""

from .memory_layout import (
    ECCCacheConfig,
    allocate_ecc_kv_cache,
    get_physical_block,
    create_block_table,
)

from .attention_ecc import (
    paged_attention_ecc,
    paged_attention_ecc_adaptive,
    reference_attention_ecc,
)

__all__ = [
    "ECCCacheConfig",
    "allocate_ecc_kv_cache",
    "get_physical_block",
    "create_block_table",
    "paged_attention_ecc",
    "paged_attention_ecc_adaptive",
    "reference_attention_ecc",
]
