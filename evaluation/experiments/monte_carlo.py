import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from ..constants import (
    CACHE_MODE_ORDER,
    CACHE_MODES,
    CACHE_MODE_LABELS,
    DEFAULT_CONFIG,
    get_cache_modes,
    get_ber_levels,
    get_seeds,
)
from ..metrics import load_wikitext2_test
from ..models import load_model
from ..sweep import SweepConfig, run_sweep, SweepResults


@dataclass
class MonteCarloConfig:
    model_name: str = "gpt2"
    cache_modes: list = None
    ber_levels: list = None
    seeds: list = None
    max_samples: int = DEFAULT_CONFIG["max_samples"]
    max_length: int = DEFAULT_CONFIG["max_length"]
    stride: int = DEFAULT_CONFIG["stride"]
    device: str = "auto"
    output_dir: str = None
    backend: str = "triton"

    compute_kl_divergence: bool = True
    compute_top5: bool = True
    compute_catastrophic: bool = True
    catastrophic_threshold: float = 1000.0

    def __post_init__(self):
        if self.cache_modes is None:
            self.cache_modes = get_cache_modes()
        if self.ber_levels is None:
            self.ber_levels = get_ber_levels(extended=True)
        if self.seeds is None:
            self.seeds = get_seeds()

    def to_sweep_config(self, clean_logits=None):
        return SweepConfig(
            cache_modes=self.cache_modes,
            ber_levels=self.ber_levels,
            seeds=self.seeds,
            max_length=self.max_length,
            stride=self.stride,
            device=self.device,
            aggregate_seeds=True,
            compute_kl_divergence=self.compute_kl_divergence,
            compute_top5=self.compute_top5,
            compute_catastrophic=self.compute_catastrophic,
            catastrophic_threshold=self.catastrophic_threshold,
            clean_logits=clean_logits,
            backend=self.backend,
        )


def run_monte_carlo_experiment(model, tokenizer, config, verbose=True):
    from ..metrics import generate_clean_logits

    texts = load_wikitext2_test(max_samples=config.max_samples)

    if verbose:
        print("Monte Carlo BER Sweep")
        print("=" * 60)
        print(f"Model: {config.model_name}")
        print(f"Cache modes: {config.cache_modes}")
        print(f"BER levels: {config.ber_levels}")
        print(f"Seeds: {config.seeds} ({len(config.seeds)} trials/config)")
        print(f"Samples: {len(texts)}")
        print(
            f"KL Divergence: {'enabled' if config.compute_kl_divergence else 'disabled'}"
        )
        print(f"Top-5 Accuracy: {'enabled' if config.compute_top5 else 'disabled'}")
        print(
            f"Catastrophic Rate: {'enabled' if config.compute_catastrophic else 'disabled'}"
        )
        print()

    clean_logits = None

    if config.compute_kl_divergence:
        if verbose:
            print("Generating clean baseline logits for KL divergence...")

        device = "cuda" if config.device == "auto" else config.device

        with torch.no_grad():
            clean_logits = generate_clean_logits(
                model,
                tokenizer,
                texts,
                max_length=config.max_length,
                device=device,
            )

        if verbose:
            print(f"  Generated {len(clean_logits)} clean logits\n")

    def progress_callback(message, current, total):
        if verbose:
            pct = (current / total) * 100 if total > 0 else 0
            print(f"  [{pct:5.1f}%] {message}")

    sweep_config = config.to_sweep_config(clean_logits=clean_logits)
    results = run_sweep(model, tokenizer, texts, sweep_config, progress_callback)

    if verbose:
        print("\nSweep complete!")

    return results


def format_results_table(results, include_std=True, include_advanced_metrics=True):
    modes = list(results.aggregated.keys())
    ber_levels = sorted(
        set(
            ber
            for mode_results in results.aggregated.values()
            for ber in mode_results.keys()
        )
    )

    headers = ["BER"] + [CACHE_MODE_LABELS.get(m, m) for m in modes]

    def build_table(title, get_mean, get_std, format_val, threshold=0.001):
        table_lines = []
        table_lines.append("")
        table_lines.append(title)
        table_lines.append("-" * 80)

        rows = []
        for ber in ber_levels:
            row = [f"{ber:.0e}" if ber > 0 else "0"]
            for mode in modes:
                if ber in results.aggregated.get(mode, {}):
                    agg = results.aggregated[mode][ber]
                    mean_val = get_mean(agg)
                    std_val = get_std(agg)
                    if include_std and std_val > threshold:
                        row.append(format_val(mean_val, std_val))
                    else:
                        row.append(format_val(mean_val, None))
                else:
                    row.append("-")
            rows.append(row)

        widths = [
            max(len(str(row[i])) for row in [headers] + rows)
            for i in range(len(headers))
        ]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        table_lines.append(header_line)
        table_lines.append("-" * len(header_line))

        for row in rows:
            table_lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))

        return table_lines

    lines = []
    lines.append("PERPLEXITY (lower is better)")
    lines.append("-" * 80)

    ppl_rows = []
    for ber in ber_levels:
        row = [f"{ber:.0e}" if ber > 0 else "0"]
        for mode in modes:
            if ber in results.aggregated.get(mode, {}):
                agg = results.aggregated[mode][ber]
                if include_std and agg.ppl_std > 0.01:
                    row.append(f"{agg.ppl_mean:.2f}+/-{agg.ppl_std:.2f}")
                else:
                    row.append(f"{agg.ppl_mean:.2f}")
            else:
                row.append("-")
        ppl_rows.append(row)

    widths = [
        max(len(str(row[i])) for row in [headers] + ppl_rows)
        for i in range(len(headers))
    ]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in ppl_rows:
        lines.append(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    if include_advanced_metrics:
        has_kl = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].kl_divergence_mean > 0
            for mode in modes
            for ber in ber_levels
        )

        if has_kl:
            lines.extend(
                build_table(
                    "KL DIVERGENCE (nats, lower is better - 0 = identical to clean)",
                    lambda agg: agg.kl_divergence_mean,
                    lambda agg: agg.kl_divergence_std,
                    lambda m, s: f"{m:.4f}+/-{s:.4f}" if s else f"{m:.4f}",
                    threshold=0.0001,
                )
            )

        has_top5 = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].top5_accuracy_mean < 1.0
            for mode in modes
            for ber in ber_levels
        )

        if has_top5:
            lines.extend(
                build_table(
                    "TOP-5 ACCURACY % (higher is better - 100% = target always in top 5)",
                    lambda agg: agg.top5_accuracy_mean * 100,
                    lambda agg: agg.top5_accuracy_std * 100,
                    lambda m, s: f"{m:.1f}+/-{s:.1f}%" if s else f"{m:.1f}%",
                    threshold=0.1,
                )
            )

        has_cat = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].catastrophic_rate_mean > 0
            for mode in modes
            for ber in ber_levels
        )

        if has_cat:
            lines.extend(
                build_table(
                    "CATASTROPHIC FAILURE RATE % (lower is better - 0% = no failures)",
                    lambda agg: agg.catastrophic_rate_mean * 100,
                    lambda agg: agg.catastrophic_rate_std * 100,
                    lambda m, s: f"{m:.1f}+/-{s:.1f}%" if s else f"{m:.1f}%",
                    threshold=0.1,
                )
            )

        has_errors = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and (
                results.aggregated[mode][ber].errors_corrected_mean > 0
                or results.aggregated[mode][ber].errors_detected_mean > 0
            )
            for mode in modes
            for ber in ber_levels
        )

        if has_errors:
            lines.append("")
            lines.append("ERROR CORRECTION STATISTICS")
            lines.append("-" * 80)

            err_headers = ["BER"] + [CACHE_MODE_LABELS.get(m, m) for m in modes]
            err_rows = []
            for ber in ber_levels:
                row = [f"{ber:.0e}" if ber > 0 else "0"]
                for mode in modes:
                    if ber in results.aggregated.get(mode, {}):
                        agg = results.aggregated[mode][ber]
                        corr = int(agg.errors_corrected_mean)
                        det = int(agg.errors_detected_mean)
                        if det > 0:
                            row.append(f"{corr:,} / {det:,}")
                        elif corr > 0:
                            row.append(f"{corr:,}")
                        else:
                            row.append("-")
                    else:
                        row.append("-")
                err_rows.append(row)

            err_widths = [
                max(len(str(row[i])) for row in [err_headers] + err_rows)
                for i in range(len(err_headers))
            ]
            err_header_line = " | ".join(
                h.ljust(w) for h, w in zip(err_headers, err_widths)
            )
            lines.append(err_header_line)
            lines.append("-" * len(err_header_line))

            for row in err_rows:
                lines.append(
                    " | ".join(str(c).ljust(w) for c, w in zip(row, err_widths))
                )

            lines.append("")
            lines.append("Note: Format is 'corrected / detected' for SECDED modes")

    return "\n".join(lines)


def format_latex_table(
    results,
    caption="Perplexity vs BER across protection strategies",
    include_advanced_metrics=True,
):
    modes = list(results.aggregated.keys())
    ber_levels = sorted(
        set(
            ber
            for mode_results in results.aggregated.values()
            for ber in mode_results.keys()
        )
    )

    latex_labels = {
        "fp16": "FP16",
        "int4": "INT4",
        "int4-hamming": "Hamming(7,4)",
        "int4-hamming84": "Hamming(8,4)",
        "int4-hamming84-interp": "H(8,4)+Interp",
        "int12-golay": "Golay(24,12)",
    }

    def build_latex_table(title, get_mean, get_std, format_val, threshold=0.001):
        table_lines = [
            "",
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{" + title + r"}",
            r"\begin{tabular}{l" + "c" * len(modes) + "}",
            r"\toprule",
            r"BER & " + " & ".join(latex_labels.get(m, m) for m in modes) + r" \\",
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

        table_lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return table_lines

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\begin{tabular}{l" + "c" * len(modes) + "}",
        r"\toprule",
        r"BER & " + " & ".join(latex_labels.get(m, m) for m in modes) + r" \\",
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

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    if include_advanced_metrics:
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

        has_errors = any(
            results.aggregated.get(mode, {}).get(ber, None) is not None
            and results.aggregated[mode][ber].errors_corrected_mean > 0
            for mode in modes
            for ber in ber_levels
        )

        if has_errors:
            lines.extend(
                [
                    "",
                    r"\begin{table}[h]",
                    r"\centering",
                    r"\caption{Error Correction Statistics vs BER}",
                    r"\begin{tabular}{l" + "c" * len(modes) + "}",
                    r"\toprule",
                    r"BER & "
                    + " & ".join(latex_labels.get(m, m) for m in modes)
                    + r" \\",
                    r"\midrule",
                ]
            )

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

            lines.extend(
                [
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{table}",
                ]
            )

    return "\n".join(lines)


def save_results(results, config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_results = {}
    for mode, mode_results in results.aggregated.items():
        json_results[mode] = {}
        for ber, agg in mode_results.items():
            json_results[mode][str(ber)] = {
                "ppl_mean": agg.ppl_mean,
                "ppl_std": agg.ppl_std,
                "kl_divergence_mean": agg.kl_divergence_mean,
                "kl_divergence_std": agg.kl_divergence_std,
                "top5_accuracy_mean": agg.top5_accuracy_mean,
                "top5_accuracy_std": agg.top5_accuracy_std,
                "catastrophic_rate_mean": agg.catastrophic_rate_mean,
                "catastrophic_rate_std": agg.catastrophic_rate_std,
                "errors_corrected_mean": agg.errors_corrected_mean,
                "errors_detected_mean": agg.errors_detected_mean,
                "n_trials": agg.n_trials,
            }

    json_path = output_path / "monte_carlo_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "results": json_results,
            },
            f,
            indent=2,
        )

    ascii_path = output_path / "results_table.txt"
    with open(ascii_path, "w") as f:
        f.write(format_results_table(results))

    latex_path = output_path / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(format_latex_table(results))

    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {ascii_path}")
    print(f"  - {latex_path}")
