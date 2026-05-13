"""
Regression gate for the TGD Part M eval.

Reads the most recent results JSON from eval/results/ and compares
against the committed baseline in eval/baseline.json.

Exits with code 1 if any metric dropped beyond its tolerance.
GitHub Actions treats exit code 1 as a failed step — the CI check
turns red and the PR cannot merge.

Usage:
    python3 eval/check_regression.py
    python3 eval/check_regression.py --results-dir eval/results --baseline eval/baseline.json
"""
import argparse
import json
import sys
from pathlib import Path


# How much each metric is allowed to drop before the build fails.
# Increase these if the gate is too sensitive; decrease if too lenient.
# MRR gets more slack because it's the most volatile metric — a single
# question's rank shifting by one position can move it noticeably.
TOLERANCES = {
    "hit_rate": 0.05,   # fail if hit rate drops more than 5 points
    "recall":   0.05,   # fail if recall drops more than 5 points
    "mrr":      0.08,   # fail if MRR drops more than 8 points
}


def load_baseline(path: str) -> dict:
    """Read the committed baseline numbers."""
    with open(path) as f:
        return json.load(f)


def load_latest_results(results_dir: str) -> dict:
    """Find the most recently written results JSON and pull out the metrics.

    Results filenames start with a timestamp (YYYYMMDD_HHMMSS_...) so
    sorting alphabetically gives the most recent file last.
    """
    results_path = Path(results_dir)
    files = sorted(results_path.glob("*.json"))

    if not files:
        print(f"ERROR: no results files found in {results_dir}")
        print("The eval must run before the regression check.")
        sys.exit(1)

    latest = files[-1]
    print(f"  Reading: {latest.name}")

    with open(latest) as f:
        data = json.load(f)

    # Pull in_scope metrics from the aggregated section of the results JSON.
    # This is the section produced by the aggregate() function in run_eval.py.
    in_scope = data["aggregated"]["in_scope"]
    return {
        "hit_rate": in_scope["hit_rate"],
        "recall":   in_scope["recall"],
        "mrr":      in_scope["mrr"],
    }


def check_regression(baseline: dict, current: dict) -> list:
    """Compare current metrics against baseline.

    Returns a list of failure messages.
    Empty list means no regression — all metrics are within tolerance.
    """
    failures = []

    for metric, tolerance in TOLERANCES.items():
        base_val = baseline[metric]
        curr_val = current[metric]
        drop = base_val - curr_val          # positive = dropped, negative = improved
        passed = drop <= tolerance

        status = "OK  " if passed else "FAIL"
        print(
            f"  {metric:<12}  baseline={base_val:.3f}  "
            f"current={curr_val:.3f}  "
            f"drop={drop:+.3f}  "
            f"tolerance={tolerance:.2f}  [{status}]"
        )

        if not passed:
            failures.append(
                f"{metric}: dropped {drop:.3f} "
                f"(baseline={base_val:.3f}, current={curr_val:.3f}, "
                f"tolerance={tolerance:.2f})"
            )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Regression gate for eval metrics.")
    parser.add_argument(
        "--results-dir",
        default="eval/results",
        help="Directory containing eval results JSON files.",
    )
    parser.add_argument(
        "--baseline",
        default="eval/baseline.json",
        help="Path to the committed baseline JSON file.",
    )
    args = parser.parse_args()

    # ---- Step 1: load baseline ------------------------------------------
    print("Baseline (committed reference):")
    baseline = load_baseline(args.baseline)
    print(
        f"  hit_rate={baseline['hit_rate']:.3f}  "
        f"recall={baseline['recall']:.3f}  "
        f"mrr={baseline['mrr']:.3f}"
    )

    # ---- Step 2: load fresh results -------------------------------------
    print("\nLatest eval results:")
    current = load_latest_results(args.results_dir)
    print(
        f"  hit_rate={current['hit_rate']:.3f}  "
        f"recall={current['recall']:.3f}  "
        f"mrr={current['mrr']:.3f}"
    )

    # ---- Step 3: compare ------------------------------------------------
    print("\nRegression check:")
    failures = check_regression(baseline, current)

    # ---- Step 4: pass or fail -------------------------------------------
    if failures:
        print(f"\nREGRESSION DETECTED — {len(failures)} metric(s) below threshold:")
        for msg in failures:
            print(f"  FAIL  {msg}")
        print(
            "\nThe eval metrics have dropped beyond the allowed tolerance.\n"
            "Options:\n"
            "  1. Fix the regression in your code.\n"
            "  2. If the new numbers are intentional (e.g. after fixing a bug\n"
            "     that was artificially inflating a metric), update\n"
            "     eval/baseline.json and explain why in the PR description."
        )
        sys.exit(1)     # exit code 1 = GitHub Actions marks the step as FAILED

    else:
        print("\nAll metrics within tolerance. No regression detected.")
        sys.exit(0)     # exit code 0 = GitHub Actions marks the step as PASSED


if __name__ == "__main__":
    main()