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
from ..latex_tables import (
    format_latex_table,
    format_storage_overhead_latex_table,
    format_throughput_latex_table,
    format_correction_rate_latex_table,
    format_error_stats_latex_table,
    format_all_latex_tables,
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
                "ppl_ci95": agg.ppl_ci95,
                "kl_divergence_mean": agg.kl_divergence_mean,
                "kl_divergence_std": agg.kl_divergence_std,
                "kl_divergence_ci95": agg.kl_divergence_ci95,
                "top5_accuracy_mean": agg.top5_accuracy_mean,
                "top5_accuracy_std": agg.top5_accuracy_std,
                "top5_accuracy_ci95": agg.top5_accuracy_ci95,
                "catastrophic_rate_mean": agg.catastrophic_rate_mean,
                "catastrophic_rate_std": agg.catastrophic_rate_std,
                "catastrophic_rate_ci95": agg.catastrophic_rate_ci95,
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

    # Single-table LaTeX (legacy format)
    latex_path = output_path / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(format_latex_table(results))

    # All tables combined for paper
    all_tables_path = output_path / "paper_tables.tex"
    with open(all_tables_path, "w") as f:
        f.write(format_all_latex_tables(results))

    # Individual table files for modular inclusion
    tables_dir = output_path / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Table 1: Perplexity
    with open(tables_dir / "perplexity.tex", "w") as f:
        f.write(format_latex_table(results, include_advanced_metrics=False))

    # Table 2: Storage Overhead
    with open(tables_dir / "storage_overhead.tex", "w") as f:
        f.write(format_storage_overhead_latex_table())

    # Table 3: Correction Rates
    corr_table = format_correction_rate_latex_table(results)
    if corr_table:
        with open(tables_dir / "correction_rates.tex", "w") as f:
            f.write(corr_table)

    # Table 4: Throughput (placeholder - needs latency_results)
    with open(tables_dir / "throughput.tex", "w") as f:
        f.write(format_throughput_latex_table(None))

    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {ascii_path}")
    print(f"  - {latex_path}")
    print(f"  - {all_tables_path}")
    print(f"  - {tables_dir}/ (individual tables)")


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo BER sweep experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model to use (gpt2, mistral-7b, llama-3.1-8b)",
    )
    parser.add_argument(
        "--cache-modes",
        type=str,
        nargs="+",
        default=None,
        help="Cache modes to test (e.g., int4 int4-hamming84 int4-hamming84-interp)",
    )
    parser.add_argument(
        "--ber-levels",
        type=float,
        nargs="+",
        default=None,
        help="BER levels to test (e.g., 0 1e-4 1e-3 1e-2)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (e.g., 42 101 997)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max WikiText-2 samples to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-kl",
        action="store_true",
        help="Disable KL divergence computation",
    )
    parser.add_argument(
        "--no-top5",
        action="store_true",
        help="Disable top-5 accuracy computation",
    )
    parser.add_argument(
        "--no-catastrophic",
        action="store_true",
        help="Disable catastrophic failure rate computation",
    )

    args = parser.parse_args()

    # Set default output dir with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{args.model}_sweep_{timestamp}"

    # Create config
    config = MonteCarloConfig(
        model_name=args.model,
        cache_modes=args.cache_modes,
        ber_levels=args.ber_levels,
        seeds=args.seeds,
        max_samples=args.max_samples,
        output_dir=args.output,
        compute_kl_divergence=not args.no_kl,
        compute_top5=not args.no_top5,
        compute_catastrophic=not args.no_catastrophic,
    )

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    # Run experiment
    results = run_monte_carlo_experiment(model, tokenizer, config)

    # Save results
    save_results(results, config, args.output)

    # Print summary table
    print("\n" + format_results_table(results))


if __name__ == "__main__":
    main()
