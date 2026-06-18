from typing import Any

from malthusjax.stats.core import TestResult, TOSTResult, StatisticalComparisonResult, StatisticalSuiteResult

def suite_to_dict(suite: StatisticalSuiteResult) -> dict[str, Any]:
    """Return a JSON-serializable representation of the suite."""
    def _test_to_dict(test: TestResult) -> dict[str, Any]:
        return {
            "name": test.name,
            "statistic": test.statistic,
            "p_value": test.p_value,
            "alternative": test.alternative,
        }

    def _tost_to_dict(tost: TOSTResult | None) -> dict[str, Any] | None:
        if tost is None:
            return None
        return {
            "margin": tost.margin,
            "lower_bound": tost.lower_bound,
            "upper_bound": tost.upper_bound,
            "t_stat_lower": tost.t_stat_lower,
            "t_stat_upper": tost.t_stat_upper,
            "p_value_lower": tost.p_value_lower,
            "p_value_upper": tost.p_value_upper,
            "p_value_max": tost.p_value_max,
            "equivalent": tost.equivalent,
        }

    def _result_to_dict(result: StatisticalComparisonResult) -> dict[str, Any]:
        return {
            "label": result.label,
            "hypothesis_text": result.hypothesis_text,
            "n_paired": result.n_paired,
            "wins_left": result.wins_left,
            "wins_right": result.wins_right,
            "ties": result.ties,
            "left_mean": result.left_mean,
            "right_mean": result.right_mean,
            "mean_diff_left_minus_right": result.mean_diff_left_minus_right,
            "median_diff_left_minus_right": result.median_diff_left_minus_right,
            "tests": {k: _test_to_dict(v) for k, v in result.tests.items()},
            "tost": _tost_to_dict(result.tost),
            "effects": {
                "cohen_dz": result.effects.cohen_dz,
                "rank_biserial": result.effects.rank_biserial,
            },
            "alpha": result.alpha,
            "decision_pass": result.decision_pass,
            "decision_basis": result.decision_basis,
            "metadata": result.metadata,
        }

    return {
        "spec": {
            "metric_name": suite.spec.metric_name,
            "hypothesis_kind": suite.spec.hypothesis_kind.value,
            "sidedness": suite.spec.sidedness.value,
            "expected_direction": suite.spec.expected_direction.value,
            "optimum_value": suite.spec.optimum_value,
            "equivalence_margin": suite.spec.equivalence_margin,
            "min_paired_seeds": suite.spec.min_paired_seeds,
            "alpha": suite.spec.alpha,
            "multiple_testing": suite.spec.multiple_testing.value,
            "include_tests": list(suite.spec.include_tests),
            "include_value_lists": suite.spec.include_value_lists,
            "include_timing_stats": suite.spec.include_timing_stats,
            "include_mean_summary": suite.spec.include_mean_summary,
        },
        "results": [_result_to_dict(r) for r in suite.results],
        "adjusted_p_values": suite.adjusted_p_values,
    }

def suite_to_markdown(suite: StatisticalSuiteResult) -> str:
    """Render suite results as a markdown summary."""
    def _fmt(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.6g}"

    lines: list[str] = []
    lines.append("# Statistical Suite Summary")
    lines.append("")
    if suite.spec.include_mean_summary:
        lines.append(
            "| Label | n | Left Start | Right Start | Left End | Right End | "
            "Diff End (L-R) | Left Delta | Right Delta | Delta Diff (L-R) | "
            "Cohen dz | Primary p | Decision | Basis |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
        )
    else:
        lines.append("| Label | n | Cohen dz | Primary p | Decision | Basis |")
        lines.append("|---|---:|---:|---:|---|---|")

    for r in suite.results:
        primary_p = None
        if r.decision_basis == "tost" and r.tost is not None:
            primary_p = r.tost.p_value_max
        elif r.decision_basis.startswith("wilcoxon") and "wilcoxon" in r.tests:
            primary_p = r.tests["wilcoxon"].p_value
        elif r.decision_basis.startswith("paired_t") and "paired_t" in r.tests:
            primary_p = r.tests["paired_t"].p_value

        p_text = "n/a" if primary_p is None else f"{primary_p:.6g}"
        if r.decision_pass is True:
            decision = "pass"
        elif r.decision_pass is False:
            decision = "fail"
        else:
            decision = "n/a"

        left_start = r.metadata.get("left_start_mean")
        right_start = r.metadata.get("right_start_mean")
        left_delta = r.metadata.get("left_end_minus_start_mean")
        right_delta = r.metadata.get("right_end_minus_start_mean")
        delta_diff = r.metadata.get("delta_end_minus_start_diff_mean")

        if suite.spec.include_mean_summary:
            lines.append(
                f"| {r.label} | {r.n_paired} | {_fmt(left_start)} | {_fmt(right_start)} | "
                f"{r.left_mean:.6g} | {r.right_mean:.6g} | "
                f"{r.mean_diff_left_minus_right:.6g} | "
                f"{_fmt(left_delta)} | {_fmt(right_delta)} | {_fmt(delta_diff)} | "
                f"{_fmt(r.effects.cohen_dz)} | {p_text} | {decision} | {r.decision_basis} |"
            )
        else:
            lines.append(
                f"| {r.label} | {r.n_paired} | {_fmt(r.effects.cohen_dz)} | "
                f"{p_text} | {decision} | {r.decision_basis} |"
            )

    for r in suite.results:
        left_start = r.metadata.get("left_start_mean")
        right_start = r.metadata.get("right_start_mean")
        left_delta = r.metadata.get("left_end_minus_start_mean")
        right_delta = r.metadata.get("right_end_minus_start_mean")
        delta_diff = r.metadata.get("delta_end_minus_start_diff_mean")
        if (
            left_start is None
            or right_start is None
            or left_delta is None
            or right_delta is None
            or delta_diff is None
        ):
            continue

        lines.append("")
        lines.append(f"## Progress Context: {r.label}")
        lines.append("")
        lines.append(
            "| Left Start | Right Start | Left Delta | Right Delta | "
            "Delta Diff (L-R) |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {_fmt(left_start)} | {_fmt(right_start)} | {_fmt(left_delta)} | "
            f"{_fmt(right_delta)} | {_fmt(delta_diff)} |"
        )

        if r.metadata.get("optimum_value") is not None:
            lines.append("")
            lines.append(
                "| Optimum | Left Start Dist | Right Start Dist | Left End Dist | "
                "Right End Dist | Left Closed % | Right Closed % | "
                "Closed %-pt Diff (L-R) |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {_fmt(r.metadata.get('optimum_value'))} | "
                f"{_fmt(r.metadata.get('left_start_distance_to_optimum'))} | "
                f"{_fmt(r.metadata.get('right_start_distance_to_optimum'))} | "
                f"{_fmt(r.metadata.get('left_end_distance_to_optimum'))} | "
                f"{_fmt(r.metadata.get('right_end_distance_to_optimum'))} | "
                f"{_fmt(r.metadata.get('left_distance_closed_pct'))} | "
                f"{_fmt(r.metadata.get('right_distance_closed_pct'))} | "
                f"{_fmt(r.metadata.get('distance_closed_pct_point_diff_left_minus_right'))} |"
            )

    if suite.adjusted_p_values:
        lines.append("")
        lines.append("## Adjusted P-values")
        lines.append("")
        lines.append("| Label | Key | Adjusted p |")
        lines.append("|---|---|---:|")
        for label, mapping in suite.adjusted_p_values.items():
            for key, value in mapping.items():
                lines.append(f"| {label} | {key} | {value:.6g} |")

    for r in suite.results:
        timing_summary = r.metadata.get("timing_summary")
        if isinstance(timing_summary, dict):
            total = timing_summary.get("duration_seconds", {})
            lines.append("")
            lines.append(f"## Timing Summary: {r.label}")
            lines.append("")
            lines.append(
                "| Timing | Left Mean | Right Mean | Left Median | Right Median | "
                "Left Min | Right Min | Left Max | Right Max | "
                "Paired Diff Mean (L-R) | Wilcoxon p |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            if isinstance(total, dict):
                lines.append(
                    f"| duration_seconds | {_fmt(total.get('left_mean'))} | "
                    f"{_fmt(total.get('right_mean'))} | "
                    f"{_fmt(total.get('left_median'))} | "
                    f"{_fmt(total.get('right_median'))} | "
                    f"{_fmt(total.get('left_min'))} | {_fmt(total.get('right_min'))} | "
                    f"{_fmt(total.get('left_max'))} | {_fmt(total.get('right_max'))} | "
                    f"{_fmt(total.get('paired_diff_mean_left_minus_right'))} | "
                    f"{_fmt(total.get('paired_wilcoxon_p_value'))} |"
                )
            components = timing_summary.get("components", {})
            if isinstance(components, dict):
                for key, comp in components.items():
                    if not isinstance(comp, dict):
                        continue
                    lines.append(
                        f"| {key} | {_fmt(comp.get('left_mean'))} | "
                        f"{_fmt(comp.get('right_mean'))} | "
                        f"{_fmt(comp.get('left_median'))} | "
                        f"{_fmt(comp.get('right_median'))} | "
                        f"{_fmt(comp.get('left_min'))} | {_fmt(comp.get('right_min'))} | "
                        f"{_fmt(comp.get('left_max'))} | {_fmt(comp.get('right_max'))} | "
                        f"{_fmt(comp.get('paired_diff_mean_left_minus_right'))} | "
                        f"{_fmt(comp.get('paired_wilcoxon_p_value'))} |"
                    )

    for r in suite.results:
        if not bool(r.metadata.get("include_value_lists", False)):
            continue
        left_vals = r.metadata.get("left_end_values")
        right_vals = r.metadata.get("right_end_values")
        if left_vals is None or right_vals is None:
            continue
        lines.append("")
        lines.append(f"## Raw Values: {r.label}")
        lines.append("")
        lines.append(f"- left_end_values: {left_vals}")
        lines.append(f"- right_end_values: {right_vals}")
        left_start_vals = r.metadata.get("left_start_values")
        right_start_vals = r.metadata.get("right_start_values")
        if left_start_vals is not None and right_start_vals is not None:
            lines.append(f"- left_start_values: {left_start_vals}")
            lines.append(f"- right_start_values: {right_start_vals}")

    return "\n".join(lines)

def regression_to_markdown(result: Any) -> str:
    """Render an OLSResult as a markdown table."""
    def _fmt(value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.6g}"

    lines: list[str] = []
    lines.append(f"# Regression Output: {result.target_name}")
    lines.append("")
    lines.append(f"**$R^2$**: {_fmt(result.r_squared)}")
    lines.append(f"**Adjusted $R^2$**: {_fmt(result.adjusted_r_squared)}")
    lines.append(f"**n**: {result.n_observations}")
    lines.append("")
    
    # Coefficients Table
    lines.append("| Feature | Coefficient | Std. Error | t-statistic | p-value |")
    lines.append("|---|---:|---:|---:|---:|")
    
    for feat in result.features:
        lines.append(
            f"| {feat} | {_fmt(result.coefficients.get(feat))} | "
            f"{_fmt(result.standard_errors.get(feat))} | "
            f"{_fmt(result.t_values.get(feat))} | "
            f"{_fmt(result.p_values.get(feat))} |"
        )
        
    if result.diagnostics:
        lines.append("")
        lines.append("## Diagnostics")
        lines.append("")
        lines.append("| Test | Statistic | p-value | Passes? |")
        lines.append("|---|---:|---:|---|")
        for diag in result.diagnostics:
            passes = "Yes" if diag.passes() else "No"
            lines.append(f"| {diag.name} | {_fmt(diag.statistic)} | {_fmt(diag.p_value)} | {passes} |")

    return "\n".join(lines)
