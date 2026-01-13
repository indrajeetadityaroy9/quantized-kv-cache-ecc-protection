"""
LaTeX Table Generation Utilities for ECC-Protected KV Cache Evaluation.

This module provides functions to generate publication-ready LaTeX tables
for research papers. It consolidates all table generation logic for:

- Perplexity vs BER sweep results
- Storage overhead comparison
- Codec throughput benchmarks
- Error correction rates and statistics
- Adaptive UEP boundary analysis
- Mode comparison tables

All functions follow a consistent style with booktabs formatting,
proper escaping, and configurable labels/captions.

Usage:
    from evaluation.latex_tables import (
        format_latex_table,
        format_storage_overhead_latex_table,
        format_all_latex_tables,
    )

    # Generate all tables for a paper
    latex_content = format_all_latex_tables(results, latency_results)
    with open("paper_tables.tex", "w") as f:
        f.write(latex_content)
"""

import math
from typing import Any, Callable, Optional

from .constants import CACHE_MODE_LABELS


# Standard LaTeX labels for cache modes
LATEX_MODE_LABELS = {
    "fp16": "FP16",
    "fp8": "FP8",
    "int4": "INT4",
    "int4-hamming": "Hamming(7,4)",
    "int4-hamming84": "Hamming(8,4)",
    "int4-hamming84-interp": "H(8,4)+Interp",
    "int12-golay": "Golay(24,12)",
}


# =============================================================================
# Core Sweep Results Tables
# =============================================================================


def format_latex_table(
    results,
    caption: str = "Perplexity vs BER across protection strategies",
    include_advanced_metrics: bool = True,
) -> str:
    """
    Generate LaTeX table for perplexity sweep results.

    This is the main results table showing perplexity (mean +/- std)
    across all cache modes and BER levels. Optionally includes
    additional metrics like KL divergence, top-5 accuracy, and
    catastrophic failure rate.

    Args:
        results: SweepResults object with aggregated field.
        caption: LaTeX table caption.
        include_advanced_metrics: Include KL, top-5, catastrophic tables.

    Returns:
        LaTeX table string with booktabs formatting.
    """
    modes = list(results.aggregated.keys())
    ber_levels = sorted(
        set(
            ber
            for mode_results in results.aggregated.values()
            for ber in mode_results.keys()
        )
    )

    def build_latex_table(title, get_mean, get_std, format_val, threshold=0.001):
        table_lines = [
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{" + title + r"}",
            r"\begin{tabular}{l" + "c" * len(modes) + "}",
            r"\toprule",
            r"BER & " + " & ".join(LATEX_MODE_LABELS.get(m, m) for m in modes) + r" \\",
            r"\midrule",
        ]

        for ber in ber_levels:
            ber_str = f"${ber:.0e}$" if ber > 0 else "0"
            values = []
            for mode in modes:
                if ber in results.aggregated.get(mode, {}):
                    agg = results.aggregated[mode][ber]
                    mean_val = get_mean(agg)
                    std_val = get_std(agg)
                    if std_val > threshold:
                        values.append(format_val(mean_val, std_val))
                    else:
                        values.append(format_val(mean_val, None))
                else:
                    values.append("-")
            table_lines.append(ber_str + " & " + " & ".join(values) + r" \\")

        table_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return table_lines

    # Main perplexity table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\begin{tabular}{l" + "c" * len(modes) + "}",
        r"\toprule",
        r"BER & " + " & ".join(LATEX_MODE_LABELS.get(m, m) for m in modes) + r" \\",
        r"\midrule",
    ]

    for ber in ber_levels:
        ber_str = f"${ber:.0e}$" if ber > 0 else "0"
        values = []
        for mode in modes:
            if ber in results.aggregated.get(mode, {}):
                agg = results.aggregated[mode][ber]
                if agg.ppl_std > 0.01:
                    values.append(f"${agg.ppl_mean:.2f} \\pm {agg.ppl_std:.2f}$")
                else:
                    values.append(f"${agg.ppl_mean:.2f}$")
            else:
                values.append("-")
        lines.append(ber_str + " & " + " & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    if include_advanced_metrics:
        # KL Divergence table
        has_kl = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].kl_divergence_mean > 0
            for mode in modes
            for ber in ber_levels
        )

        if has_kl:
            lines.extend(
                build_latex_table(
                    r"KL Divergence (nats) vs BER across protection strategies",
                    lambda agg: agg.kl_divergence_mean,
                    lambda agg: agg.kl_divergence_std,
                    lambda m, s: f"${m:.4f} \\pm {s:.4f}$" if s else f"${m:.4f}$",
                    threshold=0.0001,
                )
            )

        # Top-5 Accuracy table
        has_top5 = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].top5_accuracy_mean < 1.0
            for mode in modes
            for ber in ber_levels
        )

        if has_top5:
            lines.extend(
                build_latex_table(
                    r"Top-5 Accuracy (\%) vs BER across protection strategies",
                    lambda agg: agg.top5_accuracy_mean * 100,
                    lambda agg: agg.top5_accuracy_std * 100,
                    lambda m, s: f"${m:.1f} \\pm {s:.1f}$" if s else f"${m:.1f}$",
                    threshold=0.1,
                )
            )

        # Catastrophic Failure Rate table
        has_cat = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].catastrophic_rate_mean > 0
            for mode in modes
            for ber in ber_levels
        )

        if has_cat:
            lines.extend(
                build_latex_table(
                    r"Catastrophic Failure Rate (\%) vs BER across protection strategies",
                    lambda agg: agg.catastrophic_rate_mean * 100,
                    lambda agg: agg.catastrophic_rate_std * 100,
                    lambda m, s: f"${m:.1f} \\pm {s:.1f}$" if s else f"${m:.1f}$",
                    threshold=0.1,
                )
            )

        # Error correction statistics table
        has_errors = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].errors_corrected_mean > 0
            for mode in modes
            for ber in ber_levels
        )

        if has_errors:
            lines.extend([
                "",
                r"\begin{table}[h]",
                r"\centering",
                r"\caption{Error Correction Statistics vs BER}",
                r"\begin{tabular}{l" + "c" * len(modes) + "}",
                r"\toprule",
                r"BER & "
                + " & ".join(LATEX_MODE_LABELS.get(m, m) for m in modes)
                + r" \\",
                r"\midrule",
            ])

            for ber in ber_levels:
                ber_str = f"${ber:.0e}$" if ber > 0 else "0"
                values = []
                for mode in modes:
                    if ber in results.aggregated.get(mode, {}):
                        agg = results.aggregated[mode][ber]
                        corr = int(agg.errors_corrected_mean)
                        det = int(agg.errors_detected_mean)
                        if det > 0:
                            values.append(f"{corr:,} / {det:,}")
                        elif corr > 0:
                            values.append(f"{corr:,}")
                        else:
                            values.append("-")
                    else:
                        values.append("-")
                lines.append(ber_str + " & " + " & ".join(values) + r" \\")

            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ])

    return "\n".join(lines)


# =============================================================================
# Storage and Throughput Tables
# =============================================================================


def format_storage_overhead_latex_table() -> str:
    """
    Generate LaTeX table for storage overhead comparison.

    Compares bits per value and memory overhead across cache modes.
    Uses static data from CACHE_MODES constants.

    Returns:
        LaTeX table string for storage overhead comparison.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Storage Overhead Comparison}",
        r"\label{tab:storage-overhead}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Cache Mode & Bits/Value & Overhead vs INT4 & Overhead vs FP16 \\",
        r"\midrule",
    ]

    storage_data = [
        ("FP16 (Oracle)", 16, "+300\\%", "---"),
        ("FP8 (E4M3)", 8, "+100\\%", "$-50\\%$"),
        ("INT4 (Unprotected)", 4, "---", "$-75\\%$"),
        ("Hamming(7,4)", 7, "+75\\%", "$-56\\%$"),
        ("Hamming(8,4)", 8, "+100\\%", "$-50\\%$"),
        ("H(8,4)+Interp", 8, "+100\\%", "$-50\\%$"),
        ("Golay(24,12)", 8, "+100\\%", "$-50\\%$"),
    ]

    for name, bits, vs_int4, vs_fp16 in storage_data:
        bits_str = str(bits) if isinstance(bits, int) else bits
        lines.append(f"{name} & {bits_str} & {vs_int4} & {vs_fp16} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_throughput_latex_table(latency_results=None) -> str:
    """
    Generate LaTeX table for codec throughput comparison.

    Shows encode/decode latency, throughput in MValues/sec,
    and memory bandwidth efficiency percentage.

    Args:
        latency_results: Optional CodecBenchmarkReport from latency.py.
                        If None, uses placeholder values.

    Returns:
        LaTeX table string for throughput comparison.
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Codec Encode/Decode Throughput (A100-80GB)}",
        r"\label{tab:codec-throughput}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Codec & Encode (ms) & Decode (ms) & MVal/s & BW Eff. (\%) \\",
        r"\midrule",
    ]

    if latency_results is not None:
        # Group by codec and take representative result
        codec_results = {}
        for r in latency_results.results:
            if r.codec not in codec_results:
                codec_results[r.codec] = r
            elif r.n_values > codec_results[r.codec].n_values:
                codec_results[r.codec] = r

        codec_labels = {
            "int4": "INT4 (baseline)",
            "int4-hamming": "Hamming(7,4)",
            "int4-hamming84": "Hamming(8,4)",
            "int12-golay": "Golay(24,12)",
        }

        for codec in ["int4", "int4-hamming", "int4-hamming84", "int12-golay"]:
            if codec in codec_results:
                r = codec_results[codec]
                label = codec_labels.get(codec, codec)
                encode_str = f"${r.encode_time_mean_ms:.3f} \\pm {r.encode_time_std_ms:.3f}$"
                decode_str = f"${r.decode_time_mean_ms:.3f} \\pm {r.decode_time_std_ms:.3f}$"
                lines.append(
                    f"{label} & {encode_str} & {decode_str} & "
                    f"${r.throughput_mvalues_sec:.1f}$ & ${r.bandwidth_efficiency_pct:.1f}$ \\\\"
                )
    else:
        # No placeholder values - require actual measurements
        lines.append(
            r"\multicolumn{5}{c}{\textit{Run latency benchmark to populate this table}} \\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Error Correction Statistics Tables
# =============================================================================


def format_correction_rate_latex_table(results) -> str:
    """
    Generate LaTeX table for ECC error recovery statistics.

    Shows the fraction of detected errors that were single-bit (corrected)
    vs double-bit (detected only). For each (mode, BER) pair.
    Only includes ECC-protected modes.

    Metrics:
        Corrected (%): Fraction of error events that were single-bit and recovered
        Detected (%): Fraction of error events that were double-bit (unrecoverable)

    Args:
        results: SweepResults object with aggregated field.

    Returns:
        LaTeX table string for error recovery rates, or empty string if no data.
    """
    modes = [m for m in results.aggregated.keys() if m not in ["fp16", "fp8", "int4"]]
    ber_levels = sorted(
        set(
            ber for mode_results in results.aggregated.values()
            for ber in mode_results.keys()
        )
    )
    # Only show non-zero BER levels
    ber_levels = [b for b in ber_levels if b > 0]

    if not modes or not ber_levels:
        return ""

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Error Recovery Rates by BER}",
        r"\label{tab:correction-rates}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Codec & BER & Corrected (\%) & Detected Only (\%) \\",
        r"\midrule",
    ]

    for mode in modes:
        mode_label = LATEX_MODE_LABELS.get(mode, mode)
        first_row = True
        for ber in ber_levels:
            if ber not in results.aggregated.get(mode, {}):
                continue
            agg = results.aggregated[mode][ber]

            # Compute correction rates from raw counts or aggregated rates
            total_errors = agg.errors_corrected_mean + agg.errors_detected_mean
            if total_errors > 0:
                corr_rate = (agg.errors_corrected_mean / total_errors) * 100
                det_rate = (agg.errors_detected_mean / total_errors) * 100
            elif hasattr(agg, 'correction_rate_mean') and agg.correction_rate_mean > 0:
                corr_rate = agg.correction_rate_mean * 100
                det_rate = agg.detection_rate_mean * 100
            else:
                continue

            ber_str = f"$10^{{{int(round(math.log10(ber)))}}}$" if ber > 0 else "0"
            mode_col = mode_label if first_row else ""
            lines.append(
                f"{mode_col} & {ber_str} & ${corr_rate:.2f}$ & ${det_rate:.2f}$ \\\\"
            )
            first_row = False

        if not first_row:
            lines.append(r"\midrule")

    # Remove last midrule
    if lines[-1] == r"\midrule":
        lines.pop()

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_error_stats_latex_table(results) -> str:
    """
    Generate LaTeX table for error correction statistics (counts).

    Shows corrected/detected error counts for each (mode, BER) pair.
    Only includes protected modes and non-zero BER levels.

    Args:
        results: SweepResults object with aggregated field.

    Returns:
        LaTeX table string for error counts, or empty string if no data.
    """
    modes = list(results.aggregated.keys())
    ber_levels = sorted(
        set(
            ber for mode_results in results.aggregated.values()
            for ber in mode_results.keys()
        )
    )

    # Only include protected modes and non-zero BER
    protected_modes = [m for m in modes if m not in ["fp16", "fp8", "int4"]]
    ber_levels = [b for b in ber_levels if b > 0]

    if not protected_modes or not ber_levels:
        return ""

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Error Correction Statistics (mean counts per run)}",
        r"\label{tab:error-stats}",
        r"\begin{tabular}{l" + "c" * len(protected_modes) + "}",
        r"\toprule",
        r"BER & " + " & ".join(LATEX_MODE_LABELS.get(m, m) for m in protected_modes) + r" \\",
        r"\midrule",
    ]

    for ber in ber_levels:
        ber_str = f"$10^{{{int(round(math.log10(ber)))}}}$" if ber > 0 else "0"
        values = []
        for mode in protected_modes:
            if ber in results.aggregated.get(mode, {}):
                agg = results.aggregated[mode][ber]
                corr = int(agg.errors_corrected_mean)
                det = int(agg.errors_detected_mean)
                if det > 0:
                    values.append(f"{corr:,}/{det:,}")
                elif corr > 0:
                    values.append(f"{corr:,}")
                else:
                    values.append("---")
            else:
                values.append("---")
        lines.append(ber_str + " & " + " & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\small{Note: Format is corrected/detected for SECDED modes}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def format_single_metric_latex_table(
    results,
    title: str,
    label: str,
    get_mean: Callable[[Any], float],
    get_std: Callable[[Any], float],
    format_val: Callable[[float, Optional[float]], str],
) -> str:
    """
    Generate a single-metric LaTeX table.

    Helper function to create tables for individual metrics like
    KL divergence, top-5 accuracy, or catastrophic failure rate.

    Args:
        results: SweepResults object with aggregated field.
        title: LaTeX table caption.
        label: LaTeX table label (without 'tab:' prefix).
        get_mean: Function to extract mean value from aggregated result.
        get_std: Function to extract std value from aggregated result.
        format_val: Function to format (mean, std) as LaTeX string.

    Returns:
        LaTeX table string for the metric.
    """
    modes = list(results.aggregated.keys())
    ber_levels = sorted(
        set(
            ber for mode_results in results.aggregated.values()
            for ber in mode_results.keys()
        )
    )

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + title + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{l" + "c" * len(modes) + "}",
        r"\toprule",
        r"BER & " + " & ".join(LATEX_MODE_LABELS.get(m, m) for m in modes) + r" \\",
        r"\midrule",
    ]

    for ber in ber_levels:
        ber_str = f"$10^{{{int(round(math.log10(ber)))}}}$" if ber > 0 else "0"
        values = []
        for mode in modes:
            if ber in results.aggregated.get(mode, {}):
                agg = results.aggregated[mode][ber]
                mean_val = get_mean(agg)
                std_val = get_std(agg)
                values.append(format_val(mean_val, std_val))
            else:
                values.append("---")
        lines.append(ber_str + " & " + " & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Combined Output Functions
# =============================================================================


def format_all_latex_tables(results, latency_results=None) -> str:
    """
    Generate all LaTeX tables for the paper.

    This is the main entry point for generating a complete set of
    publication-ready tables from sweep results.

    Args:
        results: SweepResults object from monte_carlo experiment.
        latency_results: Optional CodecBenchmarkReport from latency.py.

    Returns:
        Single string containing all tables, separated by comments.
    """
    sections = []

    # Header
    sections.append("% ECC-Protected KV Cache Evaluation Tables")
    sections.append("% Generated by evaluation/latex_tables.py")
    sections.append("")

    # Table 1: Main Perplexity Results
    sections.append("% Table 1: Perplexity vs BER")
    sections.append(format_latex_table(results, include_advanced_metrics=False))
    sections.append("")

    # Table 2: Catastrophic Failure Rate
    sections.append("% Table 2: Catastrophic Failure Rate")
    cat_table = format_single_metric_latex_table(
        results,
        title=r"Catastrophic Failure Rate (\%) vs BER",
        label="tab:catastrophic-rate",
        get_mean=lambda agg: agg.catastrophic_rate_mean * 100,
        get_std=lambda agg: agg.catastrophic_rate_std * 100,
        format_val=lambda m, s: f"${m:.1f} \\pm {s:.1f}$" if s and s > 0.1 else f"${m:.1f}$",
    )
    sections.append(cat_table)
    sections.append("")

    # Table 3: KL Divergence
    sections.append("% Table 3: KL Divergence")
    kl_table = format_single_metric_latex_table(
        results,
        title=r"KL Divergence (nats) vs BER",
        label="tab:kl-divergence",
        get_mean=lambda agg: agg.kl_divergence_mean,
        get_std=lambda agg: agg.kl_divergence_std,
        format_val=lambda m, s: f"${m:.4f}$",
    )
    sections.append(kl_table)
    sections.append("")

    # Table 4: Storage Overhead
    sections.append("% Table 4: Storage Overhead")
    sections.append(format_storage_overhead_latex_table())
    sections.append("")

    # Table 5: Correction Rates
    sections.append("% Table 5: Error Correction Rates")
    corr_table = format_correction_rate_latex_table(results)
    if corr_table:
        sections.append(corr_table)
    sections.append("")

    # Table 6: Throughput (if available)
    sections.append("% Table 6: Codec Throughput")
    sections.append(format_throughput_latex_table(latency_results))
    sections.append("")

    # Table 7: Error Statistics (counts)
    sections.append("% Table 7: Error Correction Statistics (Counts)")
    err_table = format_error_stats_latex_table(results)
    if err_table:
        sections.append(err_table)

    return "\n".join(sections)
