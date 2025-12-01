#!/usr/bin/env python3
"""
Run ablation experiments across multiple domains and parameter settings.

Experiments:
1. Evaluation Set Size (|E|): [15, 25, 40]
2. Number of Candidate Questions (k): [2, 3, 5]
3. Number of Simulated Responses (t): [3, 5, 10]
4. Number of Dialogue Turns: [1, 2, 3]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import csv


# Domain configurations
DOMAIN_CONFIGS = {
    "cars": "experiments/mve_cars.yml",
    "movies": "experiments/mve_movies.yml",
    "restaurants": "experiments/mve_restaurants.yml",
}

# Fixed defaults
DEFAULTS = {
    "num_seeds": 3,
    "num_items": 15,
    "num_pairs": 25,
    "num_candidates": 3,
    "num_samples": 5,
    "num_turns": 2,
}

# Experiment definitions
EXPERIMENTS = {
    "eval_set_size": {
        "param": "num_pairs",
        "values": [15, 25, 40],
        "prefix": "E",
    },
    "num_candidates": {
        "param": "num_candidates",
        "values": [2, 3, 5],
        "prefix": "k",
    },
    "num_samples": {
        "param": "num_samples",
        "values": [3, 5, 10],
        "prefix": "t",
    },
    "num_turns": {
        "param": "num_turns",
        "values": [1, 2, 3],
        "prefix": "turns",
    },
}


def run_single_experiment(
    config_path: str,
    output_dir: Path,
    num_seeds: int,
    num_candidates: int,
    num_samples: int,
    num_pairs: int,
    num_turns: int,
    use_pllm_responder: bool = True,
    verbose: bool = False,
) -> Optional[dict]:
    """Run a single experiment configuration."""

    # Build command
    cmd = [
        sys.executable,
        "run_bestofn_verbose.py",
        "--config", config_path,
        "--num-candidates", str(num_candidates),
        "--num-samples", str(num_samples),
        "--num-seeds", str(num_seeds),
        "--output-dir", str(output_dir),
        "--no-console",  # Reduce output noise
    ]

    if use_pllm_responder:
        cmd.append("--use-pllm-responder")

    # Set num_pairs and num_turns via environment (will need to modify config loading)
    env = os.environ.copy()
    env["ABLATION_NUM_PAIRS"] = str(num_pairs)
    env["ABLATION_NUM_TURNS"] = str(num_turns)

    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            # No timeout - let runs complete naturally
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            return None

        # Load results
        aggregate_path = output_dir / "aggregate_results.json"
        if aggregate_path.exists():
            with open(aggregate_path) as f:
                return json.load(f)
        else:
            # Try to find the latest run directory
            run_dirs = list(output_dir.glob("bestofn_verbose_*"))
            if run_dirs:
                latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
                agg = latest / "aggregate_results.json"
                if agg.exists():
                    with open(agg) as f:
                        return json.load(f)

        print(f"  WARNING: No results found in {output_dir}")
        return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def is_run_complete(output_dir: Path) -> bool:
    """Check if a run has already completed."""
    # Check for aggregate_results.json in the output dir or any subdirectory
    if (output_dir / "aggregate_results.json").exists():
        return True

    run_dirs = list(output_dir.glob("bestofn_verbose_*"))
    for run_dir in run_dirs:
        if (run_dir / "aggregate_results.json").exists():
            return True

    return False


def load_existing_results(output_dir: Path) -> Optional[dict]:
    """Load results from a completed run."""
    if (output_dir / "aggregate_results.json").exists():
        with open(output_dir / "aggregate_results.json") as f:
            return json.load(f)

    run_dirs = list(output_dir.glob("bestofn_verbose_*"))
    for run_dir in run_dirs:
        agg = run_dir / "aggregate_results.json"
        if agg.exists():
            with open(agg) as f:
                return json.load(f)

    return None


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def run_ablation_experiments(
    base_output_dir: Path,
    experiments_to_run: Optional[List[str]] = None,
    domains_to_run: Optional[List[str]] = None,
    resume: bool = True,
    verbose: bool = False,
):
    """Run all ablation experiments."""

    if experiments_to_run is None:
        experiments_to_run = list(EXPERIMENTS.keys())

    if domains_to_run is None:
        domains_to_run = list(DOMAIN_CONFIGS.keys())

    # Calculate total runs
    total_runs = 0
    for exp_name in experiments_to_run:
        exp = EXPERIMENTS[exp_name]
        total_runs += len(exp["values"]) * len(domains_to_run)

    print(f"\n{'='*60}")
    print(f"ABLATION EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Experiments: {experiments_to_run}")
    print(f"Domains: {domains_to_run}")
    print(f"Seeds per run: {DEFAULTS['num_seeds']}")
    print(f"Total configurations: {total_runs}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*60}\n")

    # Track results
    all_results = []
    completed = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for exp_name in experiments_to_run:
        exp = EXPERIMENTS[exp_name]
        param_name = exp["param"]
        prefix = exp["prefix"]

        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp_name}")
        print(f"# Varying: {param_name} = {exp['values']}")
        print(f"{'#'*60}")

        for value in exp["values"]:
            value_dir = base_output_dir / exp_name / f"{prefix}_{value}"

            for domain in domains_to_run:
                config_path = DOMAIN_CONFIGS[domain]
                domain_dir = value_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)

                run_label = f"{exp_name}/{prefix}_{value}/{domain}"

                # Check if already complete
                if resume and is_run_complete(domain_dir):
                    print(f"[SKIP] {run_label} (already complete)")
                    results = load_existing_results(domain_dir)
                    if results:
                        all_results.append({
                            "experiment": exp_name,
                            "param": param_name,
                            "value": value,
                            "domain": domain,
                            "results": results,
                        })
                    skipped += 1
                    continue

                # Build params (use defaults with override)
                params = DEFAULTS.copy()
                params[param_name] = value

                print(f"\n[RUN {completed + skipped + failed + 1}/{total_runs}] {run_label}")
                print(f"  Params: k={params['num_candidates']}, t={params['num_samples']}, "
                      f"|E|={params['num_pairs']}, turns={params['num_turns']}")

                # Estimate time remaining
                if completed > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total_runs - completed - skipped - failed) * avg_time
                    print(f"  Estimated time remaining: {format_time(remaining)}")

                run_start = time.time()

                results = run_single_experiment(
                    config_path=config_path,
                    output_dir=domain_dir,
                    num_seeds=params["num_seeds"],
                    num_candidates=params["num_candidates"],
                    num_samples=params["num_samples"],
                    num_pairs=params["num_pairs"],
                    num_turns=params["num_turns"],
                    use_pllm_responder=True,
                    verbose=verbose,
                )

                run_time = time.time() - run_start

                if results:
                    print(f"  DONE in {format_time(run_time)}")
                    print(f"  Best-of-N accuracy: {results['bestofn']['mean_accuracy']:.2%}")
                    print(f"  Direct accuracy: {results['direct']['mean_accuracy']:.2%}")
                    print(f"  Improvement: {results['improvement']['mean_accuracy']:+.2%}")

                    all_results.append({
                        "experiment": exp_name,
                        "param": param_name,
                        "value": value,
                        "domain": domain,
                        "results": results,
                    })
                    completed += 1
                else:
                    failed += 1

    # Generate summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {completed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total time: {format_time(total_time)}")

    # Write summary CSV
    if all_results:
        write_summary_csv(all_results, base_output_dir / "summary.csv")
        write_summary_json(all_results, base_output_dir / "summary.json")

    return all_results


def write_summary_csv(results: List[dict], output_path: Path):
    """Write results to CSV for easy analysis."""

    fieldnames = [
        "experiment", "param", "value", "domain",
        "bestofn_accuracy_mean", "bestofn_accuracy_std",
        "direct_accuracy_mean", "direct_accuracy_std",
        "improvement_accuracy_mean", "improvement_accuracy_std",
        "bestofn_reward_mean", "direct_reward_mean",
        "num_seeds",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            res = r["results"]
            writer.writerow({
                "experiment": r["experiment"],
                "param": r["param"],
                "value": r["value"],
                "domain": r["domain"],
                "bestofn_accuracy_mean": res["bestofn"]["mean_accuracy"],
                "bestofn_accuracy_std": res["bestofn"]["std_accuracy"],
                "direct_accuracy_mean": res["direct"]["mean_accuracy"],
                "direct_accuracy_std": res["direct"]["std_accuracy"],
                "improvement_accuracy_mean": res["improvement"]["mean_accuracy"],
                "improvement_accuracy_std": res["improvement"]["std_accuracy"],
                "bestofn_reward_mean": res["bestofn"]["mean_reward"],
                "direct_reward_mean": res["direct"]["mean_reward"],
                "num_seeds": res["num_seeds"],
            })

    print(f"Summary CSV written to: {output_path}")


def write_summary_json(results: List[dict], output_path: Path):
    """Write full results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/outputs/ablations"),
        help="Base output directory"
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=list(EXPERIMENTS.keys()),
        help="Specific experiments to run (default: all)"
    )
    parser.add_argument(
        "--domains", nargs="+",
        choices=list(DOMAIN_CONFIGS.keys()),
        help="Specific domains to run (default: all)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't skip completed runs"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_ablation_experiments(
        base_output_dir=args.output_dir,
        experiments_to_run=args.experiments,
        domains_to_run=args.domains,
        resume=not args.no_resume,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
