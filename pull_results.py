#!/usr/bin/env python3
"""
Pull benchmark results from Modal volume using the Modal SDK.

This script provides a convenient way to retrieve benchmark results
without running the full modal_runner.py.

Usage:
    # List all available results
    python pull_results.py --list

    # Pull specific results
    python pull_results.py --name llama_benchmark_20251209_134313

    # Pull latest results
    python pull_results.py --latest

    # Pull to custom directory
    python pull_results.py --latest --output ./my_results

    # Print metrics to stdout only (don't save files)
    python pull_results.py --latest --print-only
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import modal
except ImportError:
    print("Error: Modal SDK not installed. Install with: pip install modal")
    sys.exit(1)


# Connect to the Modal app and volume
def get_volume():
    """Get the hamming74-results volume."""
    return modal.Volume.from_name("hamming74-results")


def list_results_from_volume() -> list:
    """List all benchmark results in the Modal volume."""
    vol = get_volume()

    results = []
    try:
        for entry in vol.listdir("/"):
            if entry.path != "/":
                name = entry.path.strip("/")
                # Check what files exist
                files = []
                try:
                    for f in vol.listdir(entry.path):
                        if f.path.endswith((".json", ".txt")):
                            files.append(Path(f.path).name)
                except Exception:
                    pass

                if files:
                    results.append({
                        "name": name,
                        "files": files,
                    })
    except Exception as e:
        print(f"Error listing volume: {e}")
        return []

    # Sort by name (which includes timestamp) descending
    results.sort(key=lambda x: x["name"], reverse=True)
    return results


def read_file_from_volume(path: str) -> str:
    """Read a file from the Modal volume."""
    vol = get_volume()

    # Read file content
    content = b""
    for chunk in vol.read_file(path):
        content += chunk

    return content.decode("utf-8")


def pull_results(run_name: str, output_dir: Path, print_only: bool = False) -> dict:
    """
    Pull benchmark results from the Modal volume.

    Args:
        run_name: Name of the benchmark run
        output_dir: Local directory to save files
        print_only: If True, only print metrics without saving

    Returns:
        Dictionary with pulled data
    """
    vol = get_volume()

    result = {"name": run_name, "files": {}}

    # List files in the run directory
    run_path = f"/{run_name}"
    files_to_read = ["results.json", "summary.txt", "metrics.txt"]

    for filename in files_to_read:
        file_path = f"{run_path}/{filename}"
        try:
            content = read_file_from_volume(file_path)
            if filename.endswith(".json"):
                result["files"][filename] = json.loads(content)
            else:
                result["files"][filename] = content
        except Exception:
            # File doesn't exist, skip
            pass

    if not result["files"]:
        print(f"Error: No files found for run '{run_name}'")
        return result

    # Save files locally if not print_only
    if not print_only:
        output_path = output_dir / run_name
        output_path.mkdir(parents=True, exist_ok=True)

        for filename, content in result["files"].items():
            file_path = output_path / filename
            if filename.endswith(".json"):
                with open(file_path, "w") as f:
                    json.dump(content, f, indent=2)
            else:
                with open(file_path, "w") as f:
                    f.write(content)

        print(f"Results saved to: {output_path}")
        print(f"Files: {', '.join(result['files'].keys())}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pull benchmark results from Modal volume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available benchmark results",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Name of specific benchmark run to pull",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Pull the most recent benchmark results",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--print-only", "-p",
        action="store_true",
        help="Print metrics to stdout without saving files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (for scripting)",
    )

    args = parser.parse_args()

    # List results
    if args.list:
        results = list_results_from_volume()

        if args.json:
            print(json.dumps(results, indent=2))
            return

        if not results:
            print("No benchmark results found in Modal volume.")
            print("Run a benchmark first: modal run modal_runner.py --benchmark --models llama")
            return

        print("\n" + "=" * 70)
        print("AVAILABLE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"\n{'Run Name':<45} | Files")
        print("-" * 70)

        for r in results:
            files = ", ".join(r["files"])
            print(f"{r['name']:<45} | {files}")

        print(f"\nTotal: {len(results)} benchmark run(s)")
        print("\nTo pull: python pull_results.py --name <run_name>")
        print("Latest:  python pull_results.py --latest")
        return

    # Determine which run to pull
    run_name = args.name
    if args.latest:
        results = list_results_from_volume()
        if not results:
            print("No benchmark results found.")
            return
        run_name = results[0]["name"]
        print(f"Latest run: {run_name}")

    if not run_name:
        parser.print_help()
        print("\nError: Specify --name, --latest, or --list")
        return

    # Pull results
    print(f"\nPulling results: {run_name}")
    result = pull_results(run_name, Path(args.output), args.print_only)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Print metrics if available
    if "metrics.txt" in result.get("files", {}):
        print("\n" + result["files"]["metrics.txt"])
    elif "summary.txt" in result.get("files", {}):
        print("\n" + result["files"]["summary.txt"])


if __name__ == "__main__":
    main()
