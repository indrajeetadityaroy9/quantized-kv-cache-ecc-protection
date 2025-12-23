# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Reed-Solomon RS(12,8) over GF(2^4) lookup table management.

The RS(12,8) code encodes 8 INT4 values (32 bits) into 12 symbols (48 bits),
correcting up to 2 symbol errors with 50% storage overhead.

GF(2^4) uses primitive polynomial x^4 + x + 1 (0x13).
Field elements: 0-15 (4 bits each).

Tables:
  - GF16_EXP[32]: Powers of primitive element alpha (alpha^i for i=0..30, cyclic)
  - GF16_LOG[16]: Discrete logarithms (log_alpha(x) for x=1..15)
  - GF16_INV[16]: Multiplicative inverses (x^(-1) for x=1..15)
  - RS_GENERATOR[5]: Generator polynomial coefficients g(x) = x^4 + g3*x^3 + g2*x^2 + g1*x + g0
"""

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

# GF(2^4) primitive polynomial: x^4 + x + 1 = 0x13
GF16_PRIMITIVE = 0x13


def _gf16_mul_slow(a: int, b: int) -> int:
    """Multiply two elements in GF(2^4) using shift-and-add.

    This is the slow reference implementation used only for table generation.
    """
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 0x10:  # If bit 4 is set, reduce by primitive polynomial
            a ^= GF16_PRIMITIVE
    return result & 0x0F


def build_gf16_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build GF(2^4) exp, log, and inverse tables.

    Returns:
        Tuple of (exp_table, log_table, inv_table):
            - exp_table: uint8 array [32] - alpha^i for i=0..31 (cyclic, extended for easy access)
            - log_table: uint8 array [16] - log_alpha(x) for x=0..15 (log[0] unused)
            - inv_table: uint8 array [16] - multiplicative inverse x^(-1) (inv[0] = 0)
    """
    exp_table = np.zeros(32, dtype=np.uint8)
    log_table = np.zeros(16, dtype=np.uint8)
    inv_table = np.zeros(16, dtype=np.uint8)

    # Generate exp/log tables using primitive element alpha = 2
    x = 1
    for i in range(15):
        exp_table[i] = x
        log_table[x] = i
        x = _gf16_mul_slow(x, 2)  # x * alpha

    # Extend exp table for easy modular access (avoids mod 15 in hot path)
    for i in range(15, 32):
        exp_table[i] = exp_table[i - 15]

    # Build inverse table: a^(-1) = a^(14) = alpha^(14 - log(a)) in GF(16)
    # Since order of multiplicative group is 15, a^15 = 1, so a^(-1) = a^14
    inv_table[0] = 0  # 0 has no inverse
    for i in range(1, 16):
        # a^(-1) = alpha^(15 - log_alpha(a)) mod 15 = alpha^(-log_alpha(a))
        inv_table[i] = exp_table[(15 - log_table[i]) % 15]

    return exp_table, log_table, inv_table


def build_rs_generator() -> np.ndarray:
    """Build RS(12,8) generator polynomial coefficients.

    Generator polynomial: g(x) = (x - alpha)(x - alpha^2)(x - alpha^3)(x - alpha^4)
    where alpha = 2 is the primitive element of GF(2^4).

    Returns:
        uint8 array [5] - coefficients [g0, g1, g2, g3, g4] where g4 = 1 (monic)
    """
    exp_table, _, _ = build_gf16_tables()

    # Roots: alpha^1=2, alpha^2=4, alpha^3=8, alpha^4=3
    roots = [exp_table[1], exp_table[2], exp_table[3], exp_table[4]]

    # Start with g(x) = 1
    # Multiply by (x - root) for each root
    # In GF(2^m), subtraction = addition = XOR, so (x - r) = (x + r)

    # g(x) = x + roots[0]
    g = [roots[0], 1]  # [g0, g1] = [root, 1]

    for r in roots[1:]:
        # Multiply g(x) by (x + r)
        # new_g[i] = g[i-1] XOR (r * g[i])
        new_g = [0] * (len(g) + 1)
        for i in range(len(g)):
            new_g[i] ^= _gf16_mul_slow(r, g[i])  # r * g[i]
            new_g[i + 1] ^= g[i]  # x * g[i]
        g = new_g

    return np.array(g, dtype=np.uint8)


# Module-level caches
_GF16_TABLES_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_RS_GENERATOR_CACHE: np.ndarray | None = None


def get_gf16_tables_numpy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get GF(2^4) tables as NumPy arrays (cached)."""
    global _GF16_TABLES_CACHE
    if _GF16_TABLES_CACHE is None:
        _GF16_TABLES_CACHE = build_gf16_tables()
    return _GF16_TABLES_CACHE


def get_rs_generator_numpy() -> np.ndarray:
    """Get RS generator polynomial as NumPy array (cached)."""
    global _RS_GENERATOR_CACHE
    if _RS_GENERATOR_CACHE is None:
        _RS_GENERATOR_CACHE = build_rs_generator()
    return _RS_GENERATOR_CACHE


@lru_cache(maxsize=8)
def get_gf16_exp_table(device: torch.device | str = "cuda") -> torch.Tensor:
    """Get GF(2^4) exp table as PyTorch tensor on specified device."""
    exp_np, _, _ = get_gf16_tables_numpy()
    return torch.from_numpy(exp_np.copy()).to(device=device, dtype=torch.uint8)


@lru_cache(maxsize=8)
def get_gf16_log_table(device: torch.device | str = "cuda") -> torch.Tensor:
    """Get GF(2^4) log table as PyTorch tensor on specified device."""
    _, log_np, _ = get_gf16_tables_numpy()
    return torch.from_numpy(log_np.copy()).to(device=device, dtype=torch.uint8)


@lru_cache(maxsize=8)
def get_gf16_inv_table(device: torch.device | str = "cuda") -> torch.Tensor:
    """Get GF(2^4) inverse table as PyTorch tensor on specified device."""
    _, _, inv_np = get_gf16_tables_numpy()
    return torch.from_numpy(inv_np.copy()).to(device=device, dtype=torch.uint8)


@lru_cache(maxsize=8)
def get_rs_gf16_tables(
    device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get all GF(2^4) tables as PyTorch tensors on specified device.

    Returns:
        Tuple of (exp_table, log_table, inv_table) as torch.Tensor.
    """
    return (
        get_gf16_exp_table(device),
        get_gf16_log_table(device),
        get_gf16_inv_table(device),
    )


def create_rs_stats_tensor(device: torch.device | str = "cuda") -> torch.Tensor:
    """Create tensor for RS error statistics.

    Layout: [no_error, corrected_1symbol, corrected_2symbol, uncorrectable]

    Returns:
        int64 tensor [4] initialized to zeros.
    """
    return torch.zeros(4, dtype=torch.int64, device=device)


def get_rs_error_stats_summary(stats: torch.Tensor) -> dict:
    """Convert RS stats tensor to human-readable dictionary.

    Args:
        stats: RS error statistics tensor [4] from create_rs_stats_tensor()

    Returns:
        Dictionary with error correction statistics.
    """
    s = stats.cpu().tolist()

    total_blocks = sum(s)
    total_corrected = s[1] + s[2]

    return {
        # Individual counts
        "rs_no_error": s[0],
        "rs_corrected_1symbol": s[1],
        "rs_corrected_2symbol": s[2],
        "rs_uncorrectable": s[3],

        # Aggregates
        "rs_total_blocks": total_blocks,
        "rs_total_corrected": total_corrected,

        # Rates (avoid division by zero)
        "rs_correction_rate": total_corrected / total_blocks if total_blocks > 0 else 0.0,
        "rs_uncorrectable_rate": s[3] / total_blocks if total_blocks > 0 else 0.0,
    }


# Verification functions for testing
def verify_gf16_tables() -> bool:
    """Verify GF(2^4) table correctness.

    Returns:
        True if all tables are correct, raises AssertionError otherwise.
    """
    exp_table, log_table, inv_table = get_gf16_tables_numpy()

    # Check exp/log are inverses
    for i in range(1, 16):
        assert exp_table[log_table[i]] == i, f"exp[log[{i}]] != {i}"

    for i in range(15):
        assert log_table[exp_table[i]] == i, f"log[exp[{i}]] != {i}"

    # Check inverse table: a * a^(-1) = 1
    for i in range(1, 16):
        product = _gf16_mul_slow(i, inv_table[i])
        assert product == 1, f"{i} * inv[{i}] = {product} != 1"

    # Check exp table periodicity
    for i in range(15):
        assert exp_table[i] == exp_table[i + 15], f"exp[{i}] != exp[{i + 15}]"

    return True


def verify_rs_generator() -> bool:
    """Verify RS generator polynomial has correct roots.

    Returns:
        True if generator polynomial is correct, raises AssertionError otherwise.
    """
    exp_table, _, _ = get_gf16_tables_numpy()
    g = get_rs_generator_numpy()

    # g(x) should evaluate to 0 at alpha^1, alpha^2, alpha^3, alpha^4
    for j in range(1, 5):
        root = exp_table[j]
        # Evaluate g(root) = g[0] + g[1]*root + g[2]*root^2 + g[3]*root^3 + g[4]*root^4
        result = 0
        power = 1
        for coef in g:
            result ^= _gf16_mul_slow(coef, power)
            power = _gf16_mul_slow(power, root)
        assert result == 0, f"g(alpha^{j}) = g({root}) = {result} != 0"

    return True


# Pre-computed table values for CUDA constant memory initialization
# These are the exact values that should be used in reed_solomon.cuh
GF16_EXP_VALUES = [
    1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9,  # alpha^0 to alpha^14
    1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9,  # Extended copy
    1, 2  # Extra for alignment
]

GF16_LOG_VALUES = [
    0,   # log(0) undefined, placeholder
    0,   # log(1) = 0
    1,   # log(2) = 1
    4,   # log(3) = 4
    2,   # log(4) = 2
    8,   # log(5) = 8
    5,   # log(6) = 5
    10,  # log(7) = 10
    3,   # log(8) = 3
    14,  # log(9) = 14
    9,   # log(10) = 9
    7,   # log(11) = 7
    6,   # log(12) = 6
    13,  # log(13) = 13
    11,  # log(14) = 11
    12,  # log(15) = 12
]

GF16_INV_VALUES = [
    0,   # inv(0) undefined
    1,   # inv(1) = 1
    9,   # inv(2) = 9
    14,  # inv(3) = 14
    13,  # inv(4) = 13
    11,  # inv(5) = 11
    7,   # inv(6) = 7
    6,   # inv(7) = 6
    15,  # inv(8) = 15
    2,   # inv(9) = 2
    12,  # inv(10) = 12
    5,   # inv(11) = 5
    10,  # inv(12) = 10
    4,   # inv(13) = 4
    3,   # inv(14) = 3
    8,   # inv(15) = 8
]

# RS(12,8) generator polynomial: g(x) = x^4 + 13x^3 + 12x^2 + 8x + 7
# g(x) = (x + α)(x + α²)(x + α³)(x + α⁴) where α = 2
# Roots: α¹=2, α²=4, α³=8, α⁴=3
# Coefficients: [g0, g1, g2, g3, g4] = [7, 8, 12, 13, 1]
RS_GENERATOR_VALUES = [7, 8, 12, 13, 1]
