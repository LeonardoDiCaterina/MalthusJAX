#!/usr/bin/env python3
"""Task-scoped ranking estimates and separate per-run Pareto plots."""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

CONDITIONS = ("native_ga", "native_ga_cosine", "evosax_open_es", "qdax_map_elites")


def estimate(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("nonfinite or empty observations")
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    half = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))) if sd is not None else None
    return {
        "n": len(x),
        "mean": mean,
        "sd": sd,
        "ordinary_95_ci": [mean - half, mean + half] if half is not None else None,
    }


def ranking(rows, expected_seeds):
    lookup = {}
    for row in rows:
        key = int(row["seed"]), row["condition"]
        if key in lookup:
            raise ValueError(f"duplicate run: {key}")
        lookup[key] = row
    expected = set(itertools.product(expected_seeds, CONDITIONS))
    usable = {
        key
        for key, row in lookup.items()
        if row.get("status") == "success"
        and isinstance(row.get("heldout_mean_return"), (int, float))
        and math.isfinite(row["heldout_mean_return"])
    }
    missing = sorted(expected - usable)
    protocol_compatible = all(
        r.get("study_mode") == "ranking" and len(r.get("heldout_episode_returns", [])) == 20
        for r in rows
    )
    complete = not missing and not (set(lookup) - expected) and protocol_compatible
    for seed in expected_seeds:
        hashes = {
            lookup[seed, c]["initial_population_sha256"] for c in CONDITIONS if (seed, c) in usable
        }
        if len(hashes) > 1:
            raise ValueError(f"initial population mismatch in seed {seed}")
    descriptives = {}
    for condition in CONDITIONS:
        valid = [lookup[s, condition] for s in expected_seeds if (s, condition) in usable]
        if valid:
            descriptives[condition] = estimate([r["heldout_mean_return"] for r in valid])
            descriptives[condition]["mean_training_setup_and_scan_seconds"] = float(
                np.mean([r["cost"]["train_end_to_end_seconds"] for r in valid])
            )
    contrasts = []
    for left, right in itertools.combinations(CONDITIONS, 2):
        seeds = [s for s in expected_seeds if (s, left) in usable and (s, right) in usable]
        differences = [
            lookup[s, left]["heldout_mean_return"] - lookup[s, right]["heldout_mean_return"]
            for s in seeds
        ]
        result = {
            "left": left,
            "right": right,
            "direction": "positive favors left",
            "estimate": estimate(differences) if differences else None,
            "raw_p": None,
            "holm_p": None,
        }
        if complete and len(seeds) >= 2 and np.std(differences, ddof=1) > 0:
            result["raw_p"] = float(stats.ttest_1samp(differences, 0).pvalue)
        contrasts.append(result)
    # Family is all six predeclared comparisons, not whichever looks favorable.
    inferential = complete and all(c["raw_p"] is not None for c in contrasts)
    if inferential:
        from statsmodels.stats.multitest import multipletests

        adjusted = multipletests([c["raw_p"] for c in contrasts], method="holm")[1]
        for result, value in zip(contrasts, adjusted):
            result["holm_p"] = float(value)
    model = None
    if inferential:
        import pandas as pd
        import statsmodels.formula.api as smf

        frame = pd.DataFrame(
            [
                {"seed": s, "condition": c, "return_value": lookup[s, c]["heldout_mean_return"]}
                for s, c in sorted(expected)
            ]
        )
        fitted = smf.ols("return_value ~ C(condition) + C(seed)", frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame.seed}
        )
        model = {
            "formula": "return_value ~ C(condition) + C(seed)",
            "covariance": "seed-clustered; finite cluster limitation applies",
            "coefficients": fitted.params.to_dict(),
            "standard_errors": fitted.bse.to_dict(),
        }
    return {
        "primary_metric": "held-out mean return per independent training run",
        "complete_family": complete,
        "ranking_protocol_compatible": protocol_compatible,
        "inferential_family_available": inferential,
        "missing_or_failed": missing,
        "descriptives": descriptives,
        "contrasts": contrasts,
        "condition_factor_model": model,
        "limits": [
            "Ordinary 95% intervals are estimation intervals, not simultaneous intervals.",
            "Two-sided tests with Holm across six contrasts; higher sample mean alone is not superiority.",
            "No equivalence, universal ranking, or ordering of statistically unresolved pairs.",
            "Training cost includes setup/scan, excludes held-out evaluation and is secondary.",
        ],
    }


def nondominated(objectives):
    x = np.asarray(objectives, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("expected finite [minus-return, complexity] vectors")
    return np.array([not np.any(np.all(x <= p, axis=1) & np.any(x < p, axis=1)) for p in x])


def phase1_estimation(directories, expected_seeds):
    groups = {}
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        for block in summary["blocks"]:
            coordinate, seed = block["block"].rsplit("_s", 1)
            for condition in block["conditions"]:
                key = coordinate, int(seed), condition
                if key in groups:
                    raise ValueError(f"duplicate phase1 observation {key}")
                groups[key] = json.loads(
                    (directory / block["block"] / f"{condition}.json").read_text()
                )
    contrasts = []
    pairs = [
        ("evosax_standard", "evosax_adapter", "best_fitness"),
        ("evosax_batched", "evosax_adapter", "best_fitness"),
        *(
            ("qdax_native", "qdax_adapter", metric)
            for metric in ("best_fitness", "coverage", "qd_score")
        ),
    ]
    for coordinate in sorted({key[0] for key in groups}):
        for left, right, metric in pairs:
            differences, missing = [], []
            for seed in expected_seeds:
                values = []
                for condition in (left, right):
                    row = groups.get((coordinate, seed, condition), {})
                    value = row.get("metrics", {}).get(metric)
                    if (
                        row.get("status") != "success"
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        values = []
                        break
                    values.append(value)
                if len(values) == 2:
                    differences.append(values[0] - values[1])
                else:
                    missing.append(seed)
            contrasts.append(
                {
                    "coordinate": coordinate,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "estimate": estimate(differences) if differences else None,
                    "missing_or_failed_seeds": missing,
                    "equivalence": "unavailable: no approved margins/family policy",
                }
            )
    return {
        "estimand": "native minus adapter reported metric, within initialization block",
        "contrasts": contrasts,
        "equivalence_claim": False,
        "limits": "Estimation only; ordinary intervals, not simultaneous intervals; inspect metric contracts before scientific claims.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, nargs="*", default=[], help="summary.json files")
    parser.add_argument(
        "--phase1",
        type=Path,
        nargs="*",
        default=[],
        help="phase1 directories containing summary.json",
    )
    parser.add_argument(
        "--phase3",
        type=Path,
        nargs="*",
        default=[],
        help="report.json files, partial reports allowed",
    )
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.phase1:
        estimates = phase1_estimation(args.phase1, args.expected_seeds)
        (args.output / "phase1_estimation.json").write_text(
            json.dumps(estimates, indent=2, allow_nan=False) + "\n"
        )
    if args.phase2:
        rows = [r for path in args.phase2 for r in json.loads(path.read_text())["runs"]]
        result = ranking(rows, args.expected_seeds)
        names = list(result["descriptives"])
        vectors = [
            [
                -result["descriptives"][n]["mean"],
                result["descriptives"][n]["mean_training_setup_and_scan_seconds"],
            ]
            for n in names
        ]
        frontier = nondominated(vectors) if vectors else []
        result["sample_mean_cost_frontier"] = [n for n, keep in zip(names, frontier) if keep]
        result["limits"].append(
            "Cost frontier uses sample means; frontier membership uncertainty is not resolved by plotted return intervals."
        )
        (args.output / "phase2_ranking.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        fig, ax = plt.subplots()
        for name, summary in result["descriptives"].items():
            x, y = summary["mean_training_setup_and_scan_seconds"], summary["mean"]
            ci = summary["ordinary_95_ci"]
            ax.errorbar(x, y, yerr=None if ci is None else [[y - ci[0]], [ci[1] - y]], fmt="o")
            ax.annotate(name, (x, y), fontsize=8)
        points = sorted(
            (
                result["descriptives"][n]["mean_training_setup_and_scan_seconds"],
                result["descriptives"][n]["mean"],
            )
            for n in result["sample_mean_cost_frontier"]
        )
        if points:
            ax.plot(*zip(*points), "--", alpha=0.5, label="Sample-mean trade-off frontier")
            ax.legend(fontsize=8)
        ax.set(
            xlabel="Mean training setup + scan seconds",
            ylabel="Held-out return",
            title="Descriptive configuration trade-off (not an evolved Pareto front)",
        )
        fig.tight_layout()
        fig.savefig(args.output / "phase2_return_cost.png", dpi=160)
        plt.close(fig)
    runs = {}
    for path in args.phase3:
        for run in json.loads(path.read_text()).get("runs", []):
            if run.get("status") != "success":
                continue
            key = run["seed"], run["topology"]
            if key in runs:
                raise ValueError(f"duplicate completed configuration {key}")
            runs[key] = run
    fronts = []
    for seed in sorted({s for s, _ in runs}):
        selected = [r for (s, _), r in runs.items() if s == seed]
        if len({r["initial_population_hash"] for r in selected}) > 1:
            raise ValueError("phase3 initialization mismatch")
        fig, ax = plt.subplots()
        for run in selected:
            heldout = run.get("heldout")
            if not heldout:
                raise ValueError("completed run lacks held-out objectives")
            objectives = np.asarray(heldout["objectives_minimize"])
            mask = nondominated(objectives)
            front = objectives[mask]
            ax.scatter(objectives[:, 1], -objectives[:, 0], alpha=0.15, s=12)
            order = np.argsort(front[:, 1])
            ax.plot(front[order, 1], -front[order, 0], "o-", label=run["topology"], ms=3)
            fronts.append(
                {
                    "seed": seed,
                    "topology": run["topology"],
                    "heldout_front_objectives": front.tolist(),
                    "nondominated_policies_including_duplicate_points": int(mask.sum()),
                }
            )
        ax.set(
            xlabel="Active stored connections",
            ylabel="Held-out mean return",
            title=f"Seed {seed}: separate held-out fronts (no pooling across runs)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / f"phase3_front_seed{seed}.png", dpi=160)
        plt.close(fig)
    if fronts:
        expected = set(
            itertools.product(
                args.expected_seeds, ("single", "ring", "bidirectional", "fully_connected")
            )
        )
        payload = {
            "runs": fronts,
            "missing_configurations": sorted(expected - set(runs)),
            "limits": "Per-run held-out fronts; no pooled or topology-superiority inference.",
        }
        (args.output / "phase3_fronts.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
    print(
        f"Analysis written to {args.output}; missing/failed runs are not silently excluded from inference."
    )


if __name__ == "__main__":
    main()
