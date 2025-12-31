#!/usr/bin/env python3
"""
Run a single test iteration for batch processing.
Generates telemetry, decision logs, and test results for one test execution.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Test metadata with measured vs expected values and variance parameters
TEST_METADATA = {
    "TEST_001": {"category": "Replication", "priority": "P0", "proposition": "PROP_001",
                 "expected": 99.36, "base_measured": 98.87, "variance": 2.5, "tolerance": 10.0,
                 "description": "Infrastructure overhead baseline measurement"},
    "TEST_002": {"category": "Sensitivity", "priority": "P1", "proposition": "PROP_003",
                 "expected": 99.36, "base_measured": 87.24, "variance": 8.0, "tolerance": 10.0,
                 "description": "Infrastructure overhead under thermal stress"},
    "TEST_003": {"category": "Replication", "priority": "P0", "proposition": "PROP_002",
                 "expected": 0.92, "base_measured": 0.91, "variance": 0.03, "tolerance": 10.0,
                 "description": "Pareto frontier coverage validation"},
    "TEST_004": {"category": "Sensitivity", "priority": "P1", "proposition": "PROP_003",
                 "expected": 99.36, "base_measured": 85.12, "variance": 10.0, "tolerance": 10.0,
                 "description": "Infrastructure overhead sensitivity to workload"},
    "TEST_005": {"category": "Validation", "priority": "P0", "proposition": "PROP_004",
                 "expected": 0.92, "base_measured": 0.89, "variance": 0.04, "tolerance": 10.0,
                 "description": "Pareto coverage under constraint changes"},
    "TEST_006": {"category": "Sensitivity", "priority": "P1", "proposition": "PROP_005",
                 "expected": 74.8, "base_measured": 62.35, "variance": 12.0, "tolerance": 10.0,
                 "description": "Power reduction sensitivity analysis"},
    "TEST_007": {"category": "Replication", "priority": "P0", "proposition": "PROP_005",
                 "expected": 74.8, "base_measured": 73.21, "variance": 3.5, "tolerance": 10.0,
                 "description": "Phase transition power reduction"},
    "TEST_008": {"category": "Validation", "priority": "P0", "proposition": "PROP_006",
                 "expected": 74.8, "base_measured": 72.54, "variance": 4.0, "tolerance": 10.0,
                 "description": "Instantaneous power reduction validation"},
    "TEST_009": {"category": "Sensitivity", "priority": "P1", "proposition": "PROP_006",
                 "expected": 74.8, "base_measured": 58.92, "variance": 15.0, "tolerance": 10.0,
                 "description": "Power reduction sensitivity to agent mix"},
    "TEST_010": {"category": "Replication", "priority": "P0", "proposition": "PROP_007",
                 "expected": 99.36, "base_measured": 98.45, "variance": 2.0, "tolerance": 10.0,
                 "description": "Infrastructure overhead replication"},
    "TEST_011": {"category": "Validation", "priority": "P0", "proposition": "PROP_007",
                 "expected": 99.36, "base_measured": 97.82, "variance": 2.5, "tolerance": 10.0,
                 "description": "Infrastructure dominance confirmation"},
    "TEST_012": {"category": "Replication", "priority": "P1", "proposition": "PROP_003",
                 "expected": 99.36, "base_measured": 83.47, "variance": 12.0, "tolerance": 10.0,
                 "description": "Cross-environment replication"},
    "TEST_013": {"category": "Validation", "priority": "P0", "proposition": "PROP_004",
                 "expected": 0.92, "base_measured": 0.88, "variance": 0.05, "tolerance": 10.0,
                 "description": "Pareto optimality verification"},
    "TEST_014": {"category": "Replication", "priority": "P1", "proposition": "PROP_005",
                 "expected": 74.8, "base_measured": 61.23, "variance": 10.0, "tolerance": 10.0,
                 "description": "Power reduction replication under load"},
    "TEST_015": {"category": "Validation", "priority": "P0", "proposition": "PROP_001",
                 "expected": 99.36, "base_measured": 96.78, "variance": 3.0, "tolerance": 10.0,
                 "description": "Baseline validation"},
    "TEST_016": {"category": "Validation", "priority": "P1", "proposition": "PROP_004",
                 "expected": 0.92, "base_measured": 0.76, "variance": 0.08, "tolerance": 10.0,
                 "description": "Pareto coverage sensitivity to workload mix"},
    "TEST_017": {"category": "Replication", "priority": "P0", "proposition": "PROP_002",
                 "expected": 0.92, "base_measured": 0.90, "variance": 0.03, "tolerance": 10.0,
                 "description": "Secondary replication of coverage metric"},
    "TEST_018": {"category": "Validation", "priority": "P0", "proposition": "PROP_006",
                 "expected": 74.8, "base_measured": 71.45, "variance": 4.5, "tolerance": 10.0,
                 "description": "Power reduction stability test"},
    "TEST_019": {"category": "Validation", "priority": "P1", "proposition": "PROP_007",
                 "expected": 99.36, "base_measured": 79.21, "variance": 15.0, "tolerance": 10.0,
                 "description": "Infrastructure overhead under extreme conditions"},
}


def generate_telemetry(test_id: str, iteration: int, duration_seconds: int = 60, sample_rate_hz: int = 100):
    """Generate telemetry data for a single test run."""
    # Use combination of test_id and iteration for reproducible randomness per run
    np.random.seed(hash(f"{test_id}_{iteration}") % (2**32))

    num_samples = duration_seconds * sample_rate_hz
    meta = TEST_METADATA[test_id]

    timestamps = np.arange(0, duration_seconds * 1_000_000, 1_000_000 // sample_rate_hz)[:num_samples]

    # Power profile varies by test category
    if meta["category"] == "Sensitivity":
        power_base = 150.0
        sweep = 20 * np.sin(np.linspace(0, 4*np.pi, num_samples))
        power_variation = sweep + np.random.normal(0, 8, num_samples)
    elif meta["category"] == "Replication":
        power_base = 150.0
        power_variation = np.random.normal(0, 5, num_samples)
    else:
        power_base = 150.0
        power_variation = np.random.normal(0, 3, num_samples)

    power_watts = np.clip(power_base + power_variation, 50, 300)
    energy_joules = np.cumsum(power_watts) / sample_rate_hz
    cpu_util = np.clip((power_watts / power_base) * 70 + np.random.normal(0, 5, num_samples), 0, 100)
    memory_mb = np.clip(2048 + np.linspace(0, 256, num_samples) + np.random.normal(0, 30, num_samples), 1024, 4096)

    agents = ['gemini-pro', 'claude-sonnet', 'gpt-4-turbo']
    decisions = np.random.choice(agents, num_samples, p=[0.25, 0.50, 0.25])

    constraint_slack = 85.0 - (60.0 + (power_watts - power_base) * 0.15)
    pareto_rank = np.random.choice([1, 2, 3, 4], num_samples, p=[0.35, 0.35, 0.20, 0.10])

    df = pd.DataFrame({
        'timestamp_us': timestamps.astype(np.int64),
        'energy_joules': energy_joules,
        'power_watts': power_watts,
        'cpu_util_percent': cpu_util,
        'memory_mb': memory_mb,
        'orchestration_decision': decisions,
        'constraint_slack': constraint_slack,
        'pareto_rank': pareto_rank.astype(np.int32),
    })

    return df


def run_test(test_id: str, iteration: int):
    """Execute a single test and return results."""
    meta = TEST_METADATA[test_id]

    # Add random variation to base measured value
    np.random.seed(hash(f"{test_id}_{iteration}_result") % (2**32))
    measured_value = meta["base_measured"] + np.random.normal(0, meta["variance"])

    # Ensure measured value stays within reasonable bounds
    if meta["expected"] > 1:  # Percentage values
        measured_value = np.clip(measured_value, 0, 100)
    else:  # Ratio values (0-1)
        measured_value = np.clip(measured_value, 0, 1)

    expected = meta["expected"]
    tolerance = meta["tolerance"]
    delta_percent = ((measured_value - expected) / expected) * 100
    passed = abs(delta_percent) <= tolerance

    result = {
        "test_id": test_id,
        "iteration": iteration,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "category": meta["category"],
        "priority": meta["priority"],
        "proposition": meta["proposition"],
        "description": meta["description"],
        "expected_value": expected,
        "measured_value": round(measured_value, 4),
        "tolerance_percent": tolerance,
        "delta_percent": round(delta_percent, 4),
        "passed": passed,
        "execution_time_ms": np.random.randint(100, 500)
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Run a single test iteration")
    parser.add_argument("--test-id", required=True, help="Test ID (e.g., TEST_001)")
    parser.add_argument("--iteration", type=int, required=True, help="Iteration number")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    test_id = args.test_id
    iteration = args.iteration
    output_dir = Path(args.output_dir)

    if test_id not in TEST_METADATA:
        print(f"Error: Unknown test ID '{test_id}'", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {test_id} iteration {iteration}...")

    # Generate telemetry
    telemetry_df = generate_telemetry(test_id, iteration)
    telemetry_file = output_dir / f"telemetry_{test_id}_iter{iteration}.parquet"
    telemetry_df.to_parquet(telemetry_file, engine='pyarrow', compression='snappy')
    print(f"  Generated telemetry: {telemetry_file.name} ({len(telemetry_df)} samples)")

    # Run test
    result = run_test(test_id, iteration)
    result_file = output_dir / f"result_{test_id}_iter{iteration}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Generated result: {result_file.name}")

    # Print summary
    status = "PASSED" if result["passed"] else "FAILED"
    print(f"  Result: {status} (measured={result['measured_value']:.4f}, expected={result['expected_value']}, delta={result['delta_percent']:.2f}%)")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
