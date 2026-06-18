import numpy as np

from malthusjax.stats.core import MultipleTestingPolicy


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = int(p.size)
    if n == 0:
        return []

    finite_mask = np.isfinite(p)
    adjusted = np.full(n, np.nan, dtype=float)
    idx = np.where(finite_mask)[0]
    if idx.size == 0:
        return adjusted.tolist()

    pf = p[idx]
    m = int(pf.size)

    order = np.argsort(pf)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    p_sorted = pf[order]

    raw = np.array([(m - i) * p_sorted[i] for i in range(m)], dtype=float)
    adj_sorted = np.maximum.accumulate(raw)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj_f = adj_sorted[inv_order]
    adjusted[idx] = adj_f
    return adjusted.tolist()


def fdr_bh(p_values: list[float]) -> list[float]:
    """Apply False Discovery Rate (Benjamini-Hochberg) correction to a list of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = int(p.size)
    if n == 0:
        return []

    finite_mask = np.isfinite(p)
    adjusted = np.full(n, np.nan, dtype=float)
    idx = np.where(finite_mask)[0]
    if idx.size == 0:
        return adjusted.tolist()

    pf = p[idx]
    m = int(pf.size)

    order = np.argsort(pf)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(m)
    p_sorted = pf[order]

    raw = np.array([m * p_sorted[i] / (i + 1) for i in range(m)], dtype=float)
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj_f = adj_sorted[inv_order]
    adjusted[idx] = adj_f
    return adjusted.tolist()


# Compatibility function
def adjust_pvalues(
    p_values: list[float],
    policy: MultipleTestingPolicy,
) -> list[float]:
    """Adjust p-values according to the requested multiple-testing policy."""
    if policy == MultipleTestingPolicy.NONE:
        return list(p_values)
    if policy == MultipleTestingPolicy.HOLM:
        return holm_bonferroni(p_values)
    if policy == MultipleTestingPolicy.FDR_BH:
        return fdr_bh(p_values)
    
    raise NotImplementedError(f"Unsupported multiple-testing policy: {policy.value}")
