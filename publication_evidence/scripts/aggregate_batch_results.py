#!/usr/bin/env python3
"""
Aggregate batch test results from multiple iterations.
Combines telemetry and test results across all runs.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def find_result_files(input_dir: Path):
    """Find all result JSON files in the input directory."""
    results = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith("result_") and file.endswith(".json"):
                results.append(Path(root) / file)
    return results


def find_telemetry_files(input_dir: Path):
    """Find all telemetry Parquet files in the input directory."""
    telemetry = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith("telemetry_") and file.endswith(".parquet"):
                telemetry.append(Path(root) / file)
    return telemetry


def aggregate_results(result_files: list):
    """Aggregate test results from all iterations."""
    all_results = []

    for result_file in result_files:
        with open(result_file, 'r') as f:
            result = json.load(f)
            all_results.append(result)

    # Group by test_id
    by_test = {}
    for result in all_results:
        test_id = result["test_id"]
        if test_id not in by_test:
            by_test[test_id] = []
        by_test[test_id].append(result)

    # Calculate statistics for each test
    aggregated = {}
    for test_id, results in by_test.items():
        measured_values = [r["measured_value"] for r in results]
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)

        aggregated[test_id] = {
            "test_id": test_id,
            "category": results[0]["category"],
            "priority": results[0]["priority"],
            "proposition": results[0]["proposition"],
            "description": results[0]["description"],
            "expected_value": results[0]["expected_value"],
            "tolerance_percent": results[0]["tolerance_percent"],
            "iterations": total_count,
            "statistics": {
                "mean": round(np.mean(measured_values), 4),
                "std": round(np.std(measured_values), 4),
                "min": round(np.min(measured_values), 4),
                "max": round(np.max(measured_values), 4),
                "median": round(np.median(measured_values), 4),
                "q25": round(np.percentile(measured_values, 25), 4),
                "q75": round(np.percentile(measured_values, 75), 4),
            },
            "pass_rate": round(passed_count / total_count * 100, 2),
            "passed_count": passed_count,
            "failed_count": total_count - passed_count,
            "all_measured_values": measured_values,
            "individual_results": results
        }

    return aggregated


def aggregate_telemetry(telemetry_files: list, output_dir: Path):
    """Aggregate telemetry data from all iterations."""
    # Group by test_id
    by_test = {}
    for tel_file in telemetry_files:
        # Extract test_id from filename: telemetry_TEST_001_iter1.parquet
        name = tel_file.stem
        parts = name.split("_")
        test_id = f"{parts[1]}_{parts[2]}"  # TEST_001
        if test_id not in by_test:
            by_test[test_id] = []
        by_test[test_id].append(tel_file)

    telemetry_stats = {}

    for test_id, files in by_test.items():
        all_power = []
        all_energy = []
        all_cpu = []

        for tel_file in files:
            df = pd.read_parquet(tel_file)
            all_power.extend(df['power_watts'].tolist())
            all_energy.append(df['energy_joules'].iloc[-1])  # Final cumulative energy
            all_cpu.extend(df['cpu_util_percent'].tolist())

        telemetry_stats[test_id] = {
            "test_id": test_id,
            "iterations": len(files),
            "total_samples": len(all_power),
            "power_stats": {
                "mean": round(np.mean(all_power), 2),
                "std": round(np.std(all_power), 2),
                "min": round(np.min(all_power), 2),
                "max": round(np.max(all_power), 2),
            },
            "energy_stats": {
                "mean_final_joules": round(np.mean(all_energy), 2),
                "std_final_joules": round(np.std(all_energy), 2),
            },
            "cpu_stats": {
                "mean": round(np.mean(all_cpu), 2),
                "std": round(np.std(all_cpu), 2),
            }
        }

    return telemetry_stats


def main():
    parser = argparse.ArgumentParser(description="Aggregate batch test results")
    parser.add_argument("--input-dir", required=True, help="Input directory with all results")
    parser.add_argument("--output-dir", required=True, help="Output directory for aggregated results")
    parser.add_argument("--iterations", type=int, default=10, help="Expected iterations per test")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating results from: {input_dir}")

    # Find all result files
    result_files = find_result_files(input_dir)
    print(f"Found {len(result_files)} result files")

    # Find all telemetry files
    telemetry_files = find_telemetry_files(input_dir)
    print(f"Found {len(telemetry_files)} telemetry files")

    # Aggregate test results
    aggregated_results = aggregate_results(result_files)
    results_file = output_dir / "aggregated_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    print(f"Saved aggregated results to: {results_file}")

    # Aggregate telemetry stats
    telemetry_stats = aggregate_telemetry(telemetry_files, output_dir)
    telemetry_file = output_dir / "aggregated_telemetry_stats.json"
    with open(telemetry_file, 'w') as f:
        json.dump(telemetry_stats, f, indent=2)
    print(f"Saved telemetry stats to: {telemetry_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATION SUMMARY")
    print("=" * 60)

    total_tests = len(aggregated_results)
    total_iterations = sum(r["iterations"] for r in aggregated_results.values())
    overall_pass_rate = sum(r["passed_count"] for r in aggregated_results.values()) / total_iterations * 100

    print(f"Total unique tests: {total_tests}")
    print(f"Total test executions: {total_iterations}")
    print(f"Overall pass rate: {overall_pass_rate:.2f}%")
    print()

    print("Per-test pass rates:")
    for test_id in sorted(aggregated_results.keys()):
        r = aggregated_results[test_id]
        status = "PASS" if r["pass_rate"] >= 50 else "FAIL"
        print(f"  {test_id}: {r['pass_rate']:.1f}% ({r['passed_count']}/{r['iterations']}) [{status}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
