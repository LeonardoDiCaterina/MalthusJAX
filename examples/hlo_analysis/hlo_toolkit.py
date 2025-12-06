"""HLO analysis toolkit

Lightweight utilities to parse and extract simple metrics from lowered HLO
text files produced by `jax.jit(...).lower(...).as_text()`.

Features:
- Count lines/chars
- Count occurrences of key IR constructs (stablehlo.while, top_k, gather, shuffle, threefry)
- Heuristic nested-loop detection (multiple stablehlo.while occurrences)
- Produce a numeric summary dict for each HLO text

This module is intentionally small and dependency-free so it can be used
inside notebooks or as a CLI helper.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List


def analyze_text(hlo_text: str) -> Dict[str, object]:
    """Analyze a single HLO text and return summary metrics.

    Args:
        hlo_text: The HLO lowered text (string).

    Returns:
        A dict with numeric metrics and simple heuristics.
    """
    metrics: Dict[str, object] = {}

    metrics["num_chars"] = len(hlo_text)
    metrics["num_lines"] = hlo_text.count("\n") + 1

    # Common patterns to inspect
    patterns = {
        "stablehlo_while": r"stablehlo\.while",
        "while": r"\bwhile\b",
        "threefry": r"threefry",
        "top_k": r"top_k|topk|TopK",
        "gather": r"gather\b",
        "shuffle": r"shuffle\b",
        "broadcast": r"broadcast\b",
        "random_split": r"random\.split|threefry|threefry2x32",
        "argmax": r"argmax\b",
        "argmin": r"argmin\b",
    }

    for name, pat in patterns.items():
        metrics[f"count_{name}"] = len(re.findall(pat, hlo_text, flags=re.IGNORECASE))

    # Heuristic: nested while loops if stablehlo.while appears >1
    metrics["heuristic_nested_while"] = metrics["count_stablehlo_while"] > 1

    # Heuristic: presence of exception handling constructs or calls that could
    # introduce control-flow (try/except may appear as 'compare', 'cond', etc.).
    metrics["count_cond"] = len(re.findall(r"\bcond\b|select\b|compare\b", hlo_text, flags=re.IGNORECASE))

    # Extract a short snippet for quick inspection (first 200 chars)
    metrics["snippet_head"] = hlo_text[:200].replace("\n", " ")
    metrics["snippet_tail"] = hlo_text[-200:].replace("\n", " ")

    return metrics


def analyze_files(paths: Iterable[str]) -> List[Dict[str, object]]:
    """Analyze multiple HLO text files.

    Args:
        paths: Iterable of filesystem paths to HLO text files.

    Returns:
        A list of summary dicts, one per file. Each dict contains a `path`
        field plus metrics from `analyze_text()`.
    """
    results: List[Dict[str, object]] = []
    for p in paths:
        try:
            with open(p, "r") as f:
                txt = f.read()
        except Exception as e:
            results.append({"path": p, "error": str(e)})
            continue

        m = analyze_text(txt)
        m["path"] = p
        results.append(m)

    return results


def summarize(results: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Produce an aggregate summary across multiple HLO analyses.

    Returns counts and means for the numeric metrics we collect.
    """
    numeric_keys = [k for k in list(next(iter(results), {}).keys()) if k.startswith("count_") or k in ("num_lines", "num_chars", "count_cond")]
    summary: Dict[str, object] = {"files": 0}
    metrics_acc = {k: 0 for k in numeric_keys}

    n_files = 0
    for r in results:
        if "error" in r:
            continue
        n_files += 1
        for k in numeric_keys:
            metrics_acc[k] += int(r.get(k, 0))

    summary["files"] = n_files
    if n_files > 0:
        for k, v in metrics_acc.items():
            summary[f"avg_{k}"] = v / n_files
    else:
        for k in numeric_keys:
            summary[f"avg_{k}"] = 0

    return summary


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: hlo_toolkit.py <hlo_file1> [<hlo_file2> ...]")
        raise SystemExit(1)

    files = sys.argv[1:]
    res = analyze_files(files)
    print(json.dumps(res, indent=2))
