# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Golay(24,12) syndrome lookup table management for hybrid ECC.

The Golay(24,12) code can correct up to 3-bit errors in a 24-bit codeword.
The syndrome table maps 12-bit syndromes (4096 entries) to correction patterns.

Table format:
  - Entry 0: No error (syndrome 0)
  - Entries 1-24: 1-bit errors
  - Entries 25-300: 2-bit errors
  - Entries 301-2324: 3-bit errors
  - Negative entries: Uncorrectable (4+ bit errors)
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

# Golay generator matrix G (12x24) - first 12 columns are identity,
# last 12 are the P matrix
GOLAY_GENERATOR_ROWS = [
    0x8009,  # Row 0 of P matrix
    0x4805,
    0x2403,
    0x1201,
    0x0989,
    0x04C5,
    0x0263,
    0x0131,
    0x0899,
    0x044D,
    0x0227,
    0x0113,
]


def _compute_syndrome(codeword: int) -> int:
    """Compute 12-bit syndrome for a 24-bit Golay codeword."""
    syndrome = 0
    for i in range(12):
        # Check parity of codeword AND with generator row
        if bin((codeword >> 12) & GOLAY_GENERATOR_ROWS[i]).count('1') % 2 == 1:
            syndrome |= (1 << i)
        if codeword & (1 << i):
            syndrome ^= GOLAY_GENERATOR_ROWS[i]
    return syndrome


def _popcount(x: int) -> int:
    """Count number of 1 bits in x."""
    return bin(x).count('1')


def _build_golay_syndrome_table() -> np.ndarray:
    """
    Build the 4096-entry syndrome lookup table for Golay(24,12).

    Returns:
        np.ndarray: int32 array of shape [4096] mapping syndromes to correction patterns.
            - Positive value: correction pattern to XOR with codeword
            - 0: No error
            - Negative: Uncorrectable error
    """
    table = np.full(4096, -1, dtype=np.int32)

    # Entry 0: No error
    table[0] = 0

    # 1-bit errors: 24 possible positions
    for i in range(24):
        error = 1 << i
        syndrome = _compute_syndrome(error)
        table[syndrome] = error

    # 2-bit errors: C(24,2) = 276 combinations
    for i in range(24):
        for j in range(i + 1, 24):
            error = (1 << i) | (1 << j)
            syndrome = _compute_syndrome(error)
            if table[syndrome] == -1:
                table[syndrome] = error

    # 3-bit errors: C(24,3) = 2024 combinations
    for i in range(24):
        for j in range(i + 1, 24):
            for k in range(j + 1, 24):
                error = (1 << i) | (1 << j) | (1 << k)
                syndrome = _compute_syndrome(error)
                if table[syndrome] == -1:
                    table[syndrome] = error

    return table


# Module-level cache for the syndrome table
_SYNDROME_TABLE_CACHE: np.ndarray | None = None


def get_syndrome_table_numpy() -> np.ndarray:
    """Get the Golay syndrome table as a NumPy array (cached)."""
    global _SYNDROME_TABLE_CACHE
    if _SYNDROME_TABLE_CACHE is None:
        _SYNDROME_TABLE_CACHE = _build_golay_syndrome_table()
    return _SYNDROME_TABLE_CACHE


@lru_cache(maxsize=8)
def get_golay_syndrome_lut(device: torch.device | str = "cuda") -> torch.Tensor:
    """
    Get the Golay syndrome lookup table as a PyTorch tensor on the specified device.

    This function is cached per device for efficient reuse.

    Args:
        device: Target device for the tensor (default: "cuda")

    Returns:
        torch.Tensor: int32 tensor of shape [4096] containing the syndrome table.
    """
    table_np = get_syndrome_table_numpy()
    return torch.from_numpy(table_np).to(device=device, dtype=torch.int32)


def create_golay_stats_tensors(
    device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create tensors for tracking Golay and Hamming error statistics.

    Returns:
        Tuple of (golay_stats, hamming_stats):
            - golay_stats: int64 tensor [5] for (no_error, corrected_1, corrected_2, corrected_3, uncorrectable)
            - hamming_stats: int64 tensor [4] for (no_error, corrected, detected, parity_only)
    """
    golay_stats = torch.zeros(5, dtype=torch.int64, device=device)
    hamming_stats = torch.zeros(4, dtype=torch.int64, device=device)
    return golay_stats, hamming_stats


def get_error_stats_summary(
    golay_stats: torch.Tensor,
    hamming_stats: torch.Tensor
) -> dict:
    """
    Convert error statistics tensors to a human-readable dictionary.

    Args:
        golay_stats: Golay error statistics tensor [5]
        hamming_stats: Hamming error statistics tensor [4]

    Returns:
        Dictionary with error correction statistics.
    """
    gs = golay_stats.cpu().tolist()
    hs = hamming_stats.cpu().tolist()

    total_golay_triplets = sum(gs)
    total_hamming_values = sum(hs)
    total_golay_corrected = gs[1] + gs[2] + gs[3]

    return {
        # Golay statistics
        "golay_no_error": gs[0],
        "golay_corrected_1bit": gs[1],
        "golay_corrected_2bit": gs[2],
        "golay_corrected_3bit": gs[3],
        "golay_uncorrectable": gs[4],
        "golay_total_triplets": total_golay_triplets,
        "golay_total_corrected": total_golay_corrected,

        # Hamming statistics
        "hamming_no_error": hs[0],
        "hamming_corrected": hs[1],
        "hamming_detected_uncorrectable": hs[2],
        "hamming_parity_only": hs[3],
        "hamming_total_values": total_hamming_values,

        # Combined statistics
        "total_corrected": total_golay_corrected + hs[1],
        "total_uncorrectable": gs[4] + hs[2],
    }
