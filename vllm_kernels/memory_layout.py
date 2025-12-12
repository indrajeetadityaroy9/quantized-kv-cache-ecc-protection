"""
Memory Layout Utilities for ECC-Protected PagedAttention.

Provides block table management and cache allocation for ECC-integrated
KV cache storage. Compatible with vLLM's PagedAttention memory layout.

vLLM Memory Layout Reference:
- key_cache: [num_blocks, num_heads, head_size/x, block_size, x]
- value_cache: [num_blocks, num_heads, head_size, block_size]
- block_table: [batch_size, max_blocks] -> physical block indices

ECC Storage Considerations:
- Hamming(8,4): uint8 (8 bits per INT4 value)
- Golay(24,12): int32 (24 bits for 3 INT4 values, padded to 32)
- Values per head dimension: hidden_size / num_heads
"""

import torch
from typing import Literal, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from hamming74.triton_kernels.config import get_physical_dtype, get_codeword_bits, get_data_bits


@dataclass
class ECCCacheConfig:
    """Configuration for ECC-protected KV cache."""

    # Model architecture
    num_heads: int
    head_size: int
    num_layers: int

    # PagedAttention config
    block_size: int = 16  # Tokens per block (vLLM default)
    num_blocks: int = 256  # Total physical blocks

    # ECC config
    codec: Literal["hamming84", "golay", "none"] = "hamming84"

    # Adaptive UEP config (optional)
    sink_blocks: int = 8  # Number of blocks to protect with stronger codec
    sink_codec: Literal["golay", "hamming84"] = "golay"
    context_codec: Literal["hamming84", "none"] = "hamming84"

    @property
    def dtype(self) -> torch.dtype:
        """Get storage dtype for the configured codec."""
        return get_physical_dtype(self.codec)

    @property
    def values_per_block(self) -> int:
        """Total INT4 values stored per block per head."""
        return self.block_size * self.head_size

    @property
    def codewords_per_block(self) -> int:
        """Number of codewords stored per block per head."""
        if self.codec == "hamming84":
            return self.values_per_block  # 1:1 mapping
        elif self.codec == "golay":
            # Golay packs 3 INT4 values per codeword
            return (self.values_per_block + 2) // 3  # Ceiling division
        else:
            return self.values_per_block

    @property
    def storage_overhead(self) -> float:
        """Storage overhead ratio compared to raw INT4."""
        if self.codec == "hamming84":
            return 8 / 4  # 8 bits vs 4 bits
        elif self.codec == "golay":
            # 32 bits (int32) for 3 * 4 = 12 bits of data
            return 32 / 12
        else:
            return 1.0


class CacheBlock(NamedTuple):
    """Reference to a physical cache block."""
    physical_idx: int
    codec: str
    dtype: torch.dtype


def allocate_ecc_kv_cache(
    config: ECCCacheConfig,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Allocate ECC-protected KV cache tensors.

    Args:
        config: Cache configuration
        device: Device to allocate on

    Returns:
        key_cache: Tensor for encoded key values
        value_cache: Tensor for encoded value values

    Note:
        Shape differs from standard vLLM to accommodate ECC codewords:
        - [num_blocks, num_layers, num_heads, codewords_per_block]
    """
    dtype = config.dtype
    shape = (
        config.num_blocks,
        config.num_layers,
        config.num_heads,
        config.codewords_per_block,
    )

    key_cache = torch.zeros(shape, dtype=dtype, device=device)
    value_cache = torch.zeros(shape, dtype=dtype, device=device)

    return key_cache, value_cache


def create_block_table(
    batch_size: int,
    max_seq_len: int,
    block_size: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a block table for PagedAttention.

    Args:
        batch_size: Number of sequences in batch
        max_seq_len: Maximum sequence length
        block_size: Tokens per block

    Returns:
        block_table: [batch_size, max_blocks] tensor of physical block indices
                     -1 indicates unallocated block
    """
    max_blocks = (max_seq_len + block_size - 1) // block_size
    block_table = torch.full(
        (batch_size, max_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    return block_table


def get_physical_block(
    block_table: torch.Tensor,
    batch_idx: int,
    logical_block_idx: int,
) -> int:
    """
    Get physical block index from block table.

    Args:
        block_table: Block table tensor
        batch_idx: Batch index
        logical_block_idx: Logical block index within sequence

    Returns:
        Physical block index, or -1 if not allocated
    """
    return int(block_table[batch_idx, logical_block_idx].item())


def allocate_blocks(
    block_table: torch.Tensor,
    batch_idx: int,
    num_blocks_needed: int,
    free_blocks: torch.Tensor,
    next_free_idx: int,
) -> int:
    """
    Allocate physical blocks for a sequence.

    Args:
        block_table: Block table to update
        batch_idx: Batch index for the sequence
        num_blocks_needed: Number of blocks to allocate
        free_blocks: List of free physical block indices
        next_free_idx: Current position in free_blocks

    Returns:
        Updated next_free_idx
    """
    for i in range(num_blocks_needed):
        if next_free_idx >= len(free_blocks):
            raise RuntimeError("Out of physical blocks")
        block_table[batch_idx, i] = free_blocks[next_free_idx]
        next_free_idx += 1
    return next_free_idx


def compute_slot_mapping(
    seq_len: int,
    block_size: int,
    block_table: torch.Tensor,
    batch_idx: int,
) -> torch.Tensor:
    """
    Compute slot mapping for a sequence.

    Maps each token position to its (physical_block, slot_within_block).

    Args:
        seq_len: Current sequence length
        block_size: Tokens per block
        block_table: Block table tensor
        batch_idx: Batch index

    Returns:
        slot_mapping: [seq_len, 2] tensor of (physical_block, slot)
    """
    device = block_table.device
    token_positions = torch.arange(seq_len, device=device)

    logical_blocks = token_positions // block_size
    slots_within_block = token_positions % block_size

    physical_blocks = block_table[batch_idx, logical_blocks]

    slot_mapping = torch.stack([physical_blocks, slots_within_block], dim=1)
    return slot_mapping


def get_codec_for_block(
    block_idx: int,
    sink_blocks: int,
    sink_codec: str = "golay",
    context_codec: str = "hamming84",
) -> str:
    """
    Determine codec for a block based on its position.

    Implements adaptive UEP: stronger codec for sink tokens (attention sinks).

    Args:
        block_idx: Logical block index in sequence
        sink_blocks: Number of blocks to protect with sink_codec
        sink_codec: Codec for sink blocks
        context_codec: Codec for remaining blocks

    Returns:
        Codec name ("golay" or "hamming84")
    """
    if block_idx < sink_blocks:
        return sink_codec
    return context_codec


# =============================================================================
# Verification
# =============================================================================

def verify_memory_layout():
    """Verify memory layout utilities work correctly."""
    print("Memory Layout Verification")
    print("=" * 60)

    # Create config
    config = ECCCacheConfig(
        num_heads=32,
        head_size=128,
        num_layers=32,
        block_size=16,
        num_blocks=256,
        codec="hamming84",
    )

    print(f"Config: {config.num_heads} heads, {config.head_size} head_size")
    print(f"Block size: {config.block_size} tokens")
    print(f"Codec: {config.codec} -> dtype={config.dtype}")
    print(f"Values per block: {config.values_per_block}")
    print(f"Codewords per block: {config.codewords_per_block}")
    print(f"Storage overhead: {config.storage_overhead:.2f}x")

    # Test with CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Allocate cache
    key_cache, value_cache = allocate_ecc_kv_cache(config, device=device)
    print(f"\nKey cache shape: {key_cache.shape}")
    print(f"Value cache shape: {value_cache.shape}")
    print(f"Total cache memory: {(key_cache.numel() + value_cache.numel()) * key_cache.element_size() / 1e6:.2f} MB")

    # Create block table
    batch_size = 4
    max_seq_len = 2048
    block_table = create_block_table(batch_size, max_seq_len, config.block_size, device=device)
    print(f"\nBlock table shape: {block_table.shape}")

    # Simulate allocation
    free_blocks = torch.arange(config.num_blocks, device=device)
    next_free = 0
    for batch_idx in range(batch_size):
        seq_len = 512 + batch_idx * 128  # Varying sequence lengths
        num_blocks_needed = (seq_len + config.block_size - 1) // config.block_size
        next_free = allocate_blocks(block_table, batch_idx, num_blocks_needed, free_blocks, next_free)
        print(f"  Batch {batch_idx}: {seq_len} tokens -> {num_blocks_needed} blocks")

    # Test slot mapping
    slot_mapping = compute_slot_mapping(100, config.block_size, block_table, batch_idx=0)
    print(f"\nSlot mapping for 100 tokens: {slot_mapping.shape}")
    print(f"  First 5 slots: {slot_mapping[:5].tolist()}")

    # Test adaptive UEP
    print("\nAdaptive UEP codec assignment:")
    for block_idx in range(10):
        codec = get_codec_for_block(block_idx, sink_blocks=4)
        print(f"  Block {block_idx}: {codec}")

    print("\nMemory layout verification passed!")
    return True


if __name__ == "__main__":
    verify_memory_layout()
