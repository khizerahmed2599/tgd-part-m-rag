import argparse
import json
import sys
from pathlib import Path
 
 
TOLERANCES = {
    "mean_faithfulness": 0.08,    # allow up to 8 points drop
    "refusal_accuracy":  0.10,    # allow up to 10 points drop
}
 
 
def load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
 
 
def load_latest_results(results_dir: str) -> dict:
    files = sorted(Path(results_dir).glob("*.json"))
    if not files:
        print(f"ERROR: no results in {results_dir}")
        sys.exit(1)
    latest = files[-1]
    print(f"  Reading: {latest.name}")
    with open(latest) as f:
        data = json.load(f)
    agg = data["aggregated"]
    return {
        "mean_faithfulness": agg.get("in_scope", {}).get("mean_faithfulness", 0.0),
        "refusal_accuracy":  agg.get("refusal", {}).get("refusal_accuracy", 0.0),
    }
 
 
def check_regression(baseline: dict, current: dict) -> list:
    failures = []
    for metric, tolerance in TOLERANCES.items():
        base = baseline[metric]
        curr = current[metric]
        drop = base - curr
        passed = drop <= tolerance
        print(
            f"  {metric:<22}  baseline={base:.3f}  "
            f"current={curr:.3f}  drop={drop:+.3f}  "
            f"tolerance={tolerance:.2f}  [{'OK  ' if passed else 'FAIL'}]"
        )
        if not passed:
            failures.append(
                f"{metric}: dropped {drop:.3f} "
                f"(baseline={base:.3f}, current={curr:.3f})"
            )
    return failures
 
 
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="eval/gen_results")
    parser.add_argument("--baseline", default="eval/gen_baseline.json")
    args = parser.parse_args()
 
    print("Baseline (generation):")
    baseline = load_baseline(args.baseline)
    print(
        f"  faithfulness={baseline['mean_faithfulness']:.3f}  "
        f"refusal_accuracy={baseline['refusal_accuracy']:.3f}"
    )
 
    print("\nLatest generation eval results:")
    current = load_latest_results(args.results_dir)
    print(
        f"  faithfulness={current['mean_faithfulness']:.3f}  "
        f"refusal_accuracy={current['refusal_accuracy']:.3f}"
    )
 
    print("\nRegression check:")
    failures = check_regression(baseline, current)
 
    if failures:
        print(f"\nREGRESSION DETECTED — {len(failures)} metric(s) below threshold:")
        for msg in failures:
            print(f"  FAIL  {msg}")
        print(
            "\nGeneration quality has regressed beyond tolerance.\n"
            "Options:\n"
            "  1. Fix the regression (prompt, model, retrieval quality).\n"
            "  2. If intentional, update eval/gen_baseline.json with an explanation."
        )
        sys.exit(1)
    else:
        print("\nAll generation metrics within tolerance.")
        sys.exit(0)
 
 
if __name__ == "__main__":
    main()