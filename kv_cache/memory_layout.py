import torch
from ecc_codecs.triton_kernels.config import get_physical_dtype


class ECCCacheConfig:
    def __init__(
        self,
        num_heads,
        head_size,
        num_layers,
        block_size=16,
        num_blocks=256,
        codec="hamming84",
        sink_blocks=8,
        sink_codec="golay",
        context_codec="hamming84",
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.codec = codec
        self.sink_blocks = sink_blocks
        self.sink_codec = sink_codec
        self.context_codec = context_codec

    @property
    def dtype(self):
        return get_physical_dtype(self.codec)

    @property
    def values_per_block(self):
        return self.block_size * self.head_size

    @property
    def codewords_per_block(self):
        if self.codec == "hamming84":
            return self.values_per_block
        elif self.codec == "golay":
            return (self.values_per_block + 2) // 3
        else:
            return self.values_per_block

    @property
    def storage_overhead(self):
        if self.codec == "hamming84":
            return 8 / 4
        elif self.codec == "golay":
            return 32 / 12
        else:
            return 1.0


class CacheBlock:
    def __init__(self, physical_idx, codec, dtype):
        self.physical_idx = physical_idx
        self.codec = codec
        self.dtype = dtype


def allocate_ecc_kv_cache(config, device="cuda"):
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


def create_block_table(batch_size, max_seq_len, block_size, device="cuda"):
    max_blocks = (max_seq_len + block_size - 1) // block_size
    block_table = torch.full(
        (batch_size, max_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    return block_table


def get_physical_block(block_table, batch_idx, logical_block_idx):
    return int(block_table[batch_idx, logical_block_idx].item())


def allocate_blocks(
    block_table,
    batch_idx,
    num_blocks_needed,
    free_blocks,
    next_free_idx,
):
    for i in range(num_blocks_needed):
        if next_free_idx >= len(free_blocks):
            raise RuntimeError("Out of physical blocks")
        block_table[batch_idx, i] = free_blocks[next_free_idx]
        next_free_idx += 1
    return next_free_idx


def compute_slot_mapping(seq_len, block_size, block_table, batch_idx):
    device = block_table.device
    token_positions = torch.arange(seq_len, device=device)

    logical_blocks = token_positions // block_size
    slots_within_block = token_positions % block_size

    physical_blocks = block_table[batch_idx, logical_blocks]

    slot_mapping = torch.stack([physical_blocks, slots_within_block], dim=1)
    return slot_mapping


def get_codec_for_block(block_idx, sink_blocks, sink_codec="golay", context_codec="hamming84"):
    if block_idx < sink_blocks:
        return sink_codec
    return context_codec


def verify_memory_layout():
    print("Memory Layout Verification")
    print("=" * 60)

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

    device = "cuda"

    key_cache, value_cache = allocate_ecc_kv_cache(config, device=device)
    print(f"\nKey cache shape: {key_cache.shape}")
    print(f"Value cache shape: {value_cache.shape}")
    print(f"Total cache memory: {(key_cache.numel() + value_cache.numel()) * key_cache.element_size() / 1e6:.2f} MB")

    batch_size = 4
    max_seq_len = 2048
    block_table = create_block_table(batch_size, max_seq_len, config.block_size, device=device)
    print(f"\nBlock table shape: {block_table.shape}")

    free_blocks = torch.arange(config.num_blocks, device=device)
    next_free = 0
    for batch_idx in range(batch_size):
        seq_len = 512 + batch_idx * 128
        num_blocks_needed = (seq_len + config.block_size - 1) // config.block_size
        next_free = allocate_blocks(
            block_table, batch_idx, num_blocks_needed, free_blocks, next_free
        )
        print(f"  Batch {batch_idx}: {seq_len} tokens -> {num_blocks_needed} blocks")

    slot_mapping = compute_slot_mapping(
        100, config.block_size, block_table, batch_idx=0
    )
    print(f"\nSlot mapping for 100 tokens: {slot_mapping.shape}")
    print(f"  First 5 slots: {slot_mapping[:5].tolist()}")

    print("\nAdaptive UEP codec assignment:")
    for block_idx in range(10):
        codec = get_codec_for_block(block_idx, sink_blocks=4)
        print(f"  Block {block_idx}: {codec}")

    print("\nMemory layout verification passed!")
    return True


if __name__ == "__main__":
    verify_memory_layout()
