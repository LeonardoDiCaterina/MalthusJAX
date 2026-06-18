import numpy as np

from malthusjax.stats.core import EffectSizeResult, PairedSample


def cohens_dz(sample: PairedSample) -> float | None:
    """Compute Cohen's dz for a paired sample."""
    if sample.n < 2:
        return float("nan")
    
    diffs = sample.diffs
    std = float(np.std(diffs, ddof=1))
    
    if np.isclose(std, 0.0):
        return 0.0
    
    return float(np.mean(diffs) / std)


def rank_biserial(sample: PairedSample) -> float | None:
    """Compute rank-biserial correlation."""
    diffs = sample.diffs
    nonzero = diffs[diffs != 0]
    
    if nonzero.size == 0:
        return None
        
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    return float((n_pos - n_neg) / (n_pos + n_neg))


def glass_delta(sample: PairedSample) -> float | None:
    """Compute Glass's Delta using the reference (right) group's standard deviation."""
    if sample.n < 2:
        return float("nan")
        
    ref_std = float(np.std(sample.right.values, ddof=1))
    
    if np.isclose(ref_std, 0.0):
        return 0.0
        
    return float(np.mean(sample.diffs) / ref_std)


# Compatibility function
def compute_effect_sizes(left: np.ndarray, right: np.ndarray) -> EffectSizeResult:
    from malthusjax.stats.core import MetricVector
    sample = PairedSample(
        left=MetricVector("left", np.asarray(left, dtype=float)),
        right=MetricVector("right", np.asarray(right, dtype=float))
    )
    return EffectSizeResult(
        cohen_dz=cohens_dz(sample),
        rank_biserial=rank_biserial(sample),
    )
