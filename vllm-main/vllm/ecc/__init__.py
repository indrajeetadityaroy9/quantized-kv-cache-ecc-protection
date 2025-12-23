# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ECC (Error Correction Code) utilities for vLLM."""

from vllm.ecc.golay_table import (
    get_golay_syndrome_lut,
    create_golay_stats_tensors,
    get_error_stats_summary,
)
from vllm.ecc.rs_tables import (
    get_rs_gf16_tables,
    create_rs_stats_tensor,
    get_rs_error_stats_summary,
)

__all__ = [
    # Golay(24,12) utilities
    "get_golay_syndrome_lut",
    "create_golay_stats_tensors",
    "get_error_stats_summary",
    # Reed-Solomon RS(12,8) utilities
    "get_rs_gf16_tables",
    "create_rs_stats_tensor",
    "get_rs_error_stats_summary",
]
