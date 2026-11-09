"""Predeclared per-seed paper inference; absent policy means descriptive only."""

import numpy as np

from .comparator import StatisticalComparator, compare_paired_arrays
from .core import (
    ExpectedDirection,
    HypothesisKind,
    MultipleTestingPolicy,
    Sidedness,
    StatisticalComparisonSpec,
    StatisticalSuiteResult,
    validate_spec,
)


def validate_policy(policy):
    for key in ("family", "sample_size_rationale", "hypotheses"):
        if not policy.get(key):
            raise ValueError(f"Statistical policy requires {key}")
    for key in ("alpha", "minimum_pairs", "multiplicity"):
        if key not in policy:
            raise ValueError(f"Statistical policy requires explicit {key}")

    specs = []
    labels = [h["label"] for h in policy["hypotheses"]]
    if len(set(labels)) != len(labels):
        raise ValueError("Hypothesis labels must be unique within a family")
    for hypothesis in policy["hypotheses"]:
        metric = hypothesis["metric"]
        if metric not in {"best_fitness", "coverage", "qd_score"}:
            raise ValueError("Unsupported archive hypothesis metric")
        kind = HypothesisKind(hypothesis["kind"])
        if kind == HypothesisKind.CLOSER_TO_OPTIMUM:
            raise ValueError(
                "Archive policies require location_shift or equivalence; optimality-distance analysis needs a declared optimum"
            )
        test = hypothesis["primary_test"]
        expected_estimand = (
            "mean_paired_difference"
            if kind == HypothesisKind.EQUIVALENCE or test == "paired_t"
            else "paired_signed_rank_location"
            if test == "wilcoxon"
            else "paired_sign_probability"
        )
        if hypothesis.get("estimand") != expected_estimand:
            raise ValueError(f"Estimand must be explicitly {expected_estimand}")
        spec = StatisticalComparisonSpec(
            metric_name=metric,
            hypothesis_kind=kind,
            sidedness=Sidedness(hypothesis["sidedness"]),
            expected_direction=ExpectedDirection(hypothesis["expected_direction"]),
            equivalence_margin=hypothesis.get("margin"),
            min_paired_seeds=policy["minimum_pairs"],
            alpha=policy["alpha"],
            multiple_testing=MultipleTestingPolicy(policy["multiplicity"]),
            include_tests=(test,),
            metric_orientation=hypothesis["metric_orientation"],
        )

        validate_spec(spec)
        specs.append(spec)
    return specs


def analyze_archive_records(left, right, policy=None, *, comparable_qd=False):
    if policy is None:
        return {
            "status": "descriptive_only",
            "reason": "No predeclared statistical policy",
            "results": [],
        }
    specs = validate_policy(policy)

    def keyed(records):
        if len({r["seed"] for r in records}) != len(records):
            raise ValueError("Duplicate inferential seed IDs")
        return {r["seed"]: r for r in records}

    ll, rr = keyed(left), keyed(right)
    if set(ll) != set(rr):
        raise ValueError("Inference requires identical paired seed sets")
    seeds = list(ll)
    results = []
    for hypothesis, spec in zip(policy["hypotheses"], specs):
        metric = spec.metric_name
        if metric == "qd_score" and not comparable_qd:
            raise ValueError("QD inference requires matching versioned metric contracts")
        for row in [*ll.values(), *rr.values()]:
            contract = row.get("metric_metadata", {})
            declared = contract.get("metrics", {}).get(metric, {}).get("orientation")
            if declared and declared != spec.metric_orientation:
                raise ValueError("Statistical metric orientation conflicts with producer contract")
        expected_estimand = hypothesis["estimand"]

        def values(rows):
            return np.asarray([rows[s][metric][-1] for s in seeds], dtype=float)

        results.append(
            compare_paired_arrays(
                label=hypothesis["label"],
                left_name="malthusjax_native",
                right_name="qdax_baseline",
                left=values(ll),
                right=values(rr),
                spec=spec,
                metadata={
                    "seeds": seeds,
                    "family": policy["family"],
                    "estimand_declaration": expected_estimand,
                    "sample_size_rationale": policy["sample_size_rationale"],
                },
            )
        )
    if not specs:
        raise ValueError("Inference family must contain at least one hypothesis")
    suite = StatisticalSuiteResult(spec=specs[0], results=results)
    StatisticalComparator().adjust_suite_pvalues(suite)
    payload = suite.to_dict()
    payload.update(
        status="valid" if all(r.status == "valid" for r in results) else "unavailable",
        policy=policy,
        scope="seed-level inference; no power certification implied",
    )
    return payload


def inference_text(analysis, metric):
    matches = [
        r for r in analysis.get("results", []) if r.get("metadata", {}).get("metric_name") == metric
    ]
    if not matches:
        return "Descriptive only (no inference policy)"
    texts = []
    for result in matches:
        p = (
            result.get("adjusted_p_value")
            if result.get("decision_scope") == "family_corrected"
            else next(
                (
                    t["p_value"]
                    for name, t in result["tests"].items()
                    if result["decision_basis"].startswith(name)
                ),
                None,
            )
        )
        ptext = "unavailable" if p is None else f"p={p:.3g}"
        texts.append(
            f"{result['conclusion'].replace('_', ' ')}; {ptext} ({result['decision_scope'].replace('_', ' ')})"
        )
    return "; ".join(texts)
