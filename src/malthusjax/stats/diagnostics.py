import numpy as np

from malthusjax.stats.core import DiagnosticResult

def breusch_pagan(resid: np.ndarray, exog: np.ndarray) -> DiagnosticResult:
    """Perform the Breusch-Pagan test for heteroskedasticity.
    
    Returns a DiagnosticResult with the LM test p-value.
    A p-value < 0.05 indicates the presence of heteroskedasticity.
    """
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        _, pval, _, _ = het_breuschpagan(resid, exog)
        return DiagnosticResult(
            name="breusch_pagan",
            statistic=None,
            p_value=float(pval),
        )
    except Exception:
        return DiagnosticResult(
            name="breusch_pagan",
            statistic=None,
            p_value=None,
        )

def shapiro_wilk(resid: np.ndarray, max_samples: int = 5000) -> DiagnosticResult:
    """Perform the Shapiro-Wilk test for normality of residuals.
    
    If len(resid) > max_samples, a random subset is used.
    A p-value < 0.05 indicates the residuals are not normally distributed.
    """
    import scipy.stats as stats
    
    if len(resid) == 0:
        return DiagnosticResult("shapiro_wilk", None, None)
        
    arr = np.asarray(resid)
    if len(arr) > max_samples:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, max_samples, replace=False)
        
    try:
        stat, pval = stats.shapiro(arr)
        return DiagnosticResult(
            name="shapiro_wilk",
            statistic=float(stat),
            p_value=float(pval),
        )
    except Exception:
        return DiagnosticResult(
            name="shapiro_wilk",
            statistic=None,
            p_value=None,
        )
