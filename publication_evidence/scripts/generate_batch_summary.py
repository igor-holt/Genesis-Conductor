#!/usr/bin/env python3
"""
Generate final summary report from aggregated batch test results.
Creates a publication-ready summary with statistical analysis.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def load_aggregated_results(input_dir: Path):
    """Load aggregated test results."""
    results_file = input_dir / "aggregated_test_results.json"
    if not results_file.exists():
        return None
    with open(results_file, 'r') as f:
        return json.load(f)


def load_telemetry_stats(input_dir: Path):
    """Load aggregated telemetry statistics."""
    telemetry_file = input_dir / "aggregated_telemetry_stats.json"
    if not telemetry_file.exists():
        return None
    with open(telemetry_file, 'r') as f:
        return json.load(f)


def calculate_confidence_interval(values, confidence=0.95):
    """Calculate confidence interval for a list of values."""
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    # Using t-distribution approximation for small samples
    from scipy import stats
    try:
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * std / np.sqrt(n)
        return {
            "mean": round(mean, 4),
            "lower": round(mean - margin, 4),
            "upper": round(mean + margin, 4),
            "confidence": confidence
        }
    except:
        # Fallback without scipy
        margin = 1.96 * std / np.sqrt(n)  # Approximate with z-score
        return {
            "mean": round(mean, 4),
            "lower": round(mean - margin, 4),
            "upper": round(mean + margin, 4),
            "confidence": confidence
        }


def generate_summary(aggregated_results: dict, telemetry_stats: dict):
    """Generate comprehensive summary report."""

    # Calculate overall statistics
    total_tests = len(aggregated_results)
    total_executions = sum(r["iterations"] for r in aggregated_results.values())
    total_passed = sum(r["passed_count"] for r in aggregated_results.values())
    total_failed = total_executions - total_passed
    overall_pass_rate = total_passed / total_executions * 100

    # Group by category
    by_category = {}
    for test_id, result in aggregated_results.items():
        cat = result["category"]
        if cat not in by_category:
            by_category[cat] = {"tests": [], "passed": 0, "total": 0}
        by_category[cat]["tests"].append(test_id)
        by_category[cat]["passed"] += result["passed_count"]
        by_category[cat]["total"] += result["iterations"]

    category_stats = {}
    for cat, data in by_category.items():
        category_stats[cat] = {
            "test_count": len(data["tests"]),
            "total_executions": data["total"],
            "passed": data["passed"],
            "pass_rate": round(data["passed"] / data["total"] * 100, 2)
        }

    # Generate per-test results with confidence intervals
    test_results = []
    for test_id in sorted(aggregated_results.keys()):
        r = aggregated_results[test_id]
        values = r["all_measured_values"]

        try:
            ci = calculate_confidence_interval(values)
        except:
            ci = {"mean": r["statistics"]["mean"], "lower": 0, "upper": 0, "confidence": 0.95}

        test_results.append({
            "test_id": test_id,
            "category": r["category"],
            "proposition": r["proposition"],
            "expected_value": r["expected_value"],
            "mean_value": r["statistics"]["mean"],
            "std_dev": r["statistics"]["std"],
            "min_value": r["statistics"]["min"],
            "max_value": r["statistics"]["max"],
            "confidence_interval": ci,
            "pass_rate": r["pass_rate"],
            "iterations": r["iterations"],
            "verdict": "CONSISTENT_PASS" if r["pass_rate"] >= 80 else (
                "MARGINAL" if r["pass_rate"] >= 50 else "CONSISTENT_FAIL"
            )
        })

    # Identify consistently passing and failing tests
    consistent_pass = [t["test_id"] for t in test_results if t["verdict"] == "CONSISTENT_PASS"]
    marginal = [t["test_id"] for t in test_results if t["verdict"] == "MARGINAL"]
    consistent_fail = [t["test_id"] for t in test_results if t["verdict"] == "CONSISTENT_FAIL"]

    summary = {
        "report_timestamp": datetime.utcnow().isoformat() + "Z",
        "report_type": "batch_test_summary",
        "total_tests": total_tests,
        "iterations_per_test": aggregated_results[list(aggregated_results.keys())[0]]["iterations"] if aggregated_results else 0,
        "total_executions": total_executions,
        "overall_statistics": {
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": round(overall_pass_rate, 2)
        },
        "category_breakdown": category_stats,
        "test_results": test_results,
        "verdict_summary": {
            "consistent_pass": consistent_pass,
            "marginal": marginal,
            "consistent_fail": consistent_fail,
            "consistent_pass_count": len(consistent_pass),
            "marginal_count": len(marginal),
            "consistent_fail_count": len(consistent_fail)
        },
        "reproducibility_assessment": {
            "high_reproducibility": [t["test_id"] for t in test_results if t["std_dev"] < 2.0],
            "medium_reproducibility": [t["test_id"] for t in test_results if 2.0 <= t["std_dev"] < 5.0],
            "low_reproducibility": [t["test_id"] for t in test_results if t["std_dev"] >= 5.0]
        },
        "publication_recommendations": {
            "strong_evidence": consistent_pass,
            "requires_documentation": marginal,
            "known_limitations": consistent_fail,
            "overall_status": "ready_for_publication" if len(consistent_pass) >= total_tests * 0.5 else "requires_revision"
        }
    }

    # Add telemetry summary if available
    if telemetry_stats:
        summary["telemetry_summary"] = {
            "tests_with_telemetry": len(telemetry_stats),
            "average_power_watts": round(
                np.mean([t["power_stats"]["mean"] for t in telemetry_stats.values()]), 2
            ),
            "power_variance": round(
                np.mean([t["power_stats"]["std"] for t in telemetry_stats.values()]), 2
            )
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate batch test summary report")
    parser.add_argument("--input-dir", required=True, help="Input directory with aggregated results")
    parser.add_argument("--output-file", required=True, help="Output file for summary report")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    print(f"Loading aggregated results from: {input_dir}")

    # Load data
    aggregated_results = load_aggregated_results(input_dir)
    if not aggregated_results:
        print("Error: Could not load aggregated results", file=sys.stderr)
        sys.exit(1)

    telemetry_stats = load_telemetry_stats(input_dir)

    # Generate summary
    summary = generate_summary(aggregated_results, telemetry_stats)

    # Save summary
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary report saved to: {output_file}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("BATCH TEST SUMMARY REPORT")
    print("=" * 70)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Iterations per Test: {summary['iterations_per_test']}")
    print(f"Total Executions: {summary['total_executions']}")
    print(f"Overall Pass Rate: {summary['overall_statistics']['pass_rate']}%")
    print()
    print("Category Breakdown:")
    for cat, stats in summary['category_breakdown'].items():
        print(f"  {cat}: {stats['pass_rate']}% ({stats['passed']}/{stats['total_executions']})")
    print()
    print("Verdict Summary:")
    print(f"  Consistent Pass (≥80%): {summary['verdict_summary']['consistent_pass_count']} tests")
    print(f"  Marginal (50-79%): {summary['verdict_summary']['marginal_count']} tests")
    print(f"  Consistent Fail (<50%): {summary['verdict_summary']['consistent_fail_count']} tests")
    print()
    print(f"Publication Status: {summary['publication_recommendations']['overall_status'].upper()}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
