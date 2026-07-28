from typing import Callable

_TEST_REGISTRY: dict[str, Callable] = {}
_EFFECT_REGISTRY: dict[str, Callable] = {}
_DIAGNOSTIC_REGISTRY: dict[str, Callable] = {}
_CORRECTION_REGISTRY: dict[str, Callable] = {}


def register_test(name: str, fn: Callable) -> None:
    _TEST_REGISTRY[name] = fn


def get_test(name: str) -> Callable:
    if name not in _TEST_REGISTRY:
        raise KeyError(f"Unknown test: '{name}'. Available: {list(_TEST_REGISTRY)}")
    return _TEST_REGISTRY[name]


def register_effect(name: str, fn: Callable) -> None:
    _EFFECT_REGISTRY[name] = fn


def get_effect(name: str) -> Callable:
    if name not in _EFFECT_REGISTRY:
        raise KeyError(f"Unknown effect: '{name}'. Available: {list(_EFFECT_REGISTRY)}")
    return _EFFECT_REGISTRY[name]


def register_diagnostic(name: str, fn: Callable) -> None:
    _DIAGNOSTIC_REGISTRY[name] = fn


def get_diagnostic(name: str) -> Callable:
    if name not in _DIAGNOSTIC_REGISTRY:
        raise KeyError(f"Unknown diagnostic: '{name}'. Available: {list(_DIAGNOSTIC_REGISTRY)}")
    return _DIAGNOSTIC_REGISTRY[name]


def register_correction(name: str, fn: Callable) -> None:
    _CORRECTION_REGISTRY[name] = fn


def get_correction(name: str) -> Callable:
    if name not in _CORRECTION_REGISTRY:
        raise KeyError(f"Unknown correction: '{name}'. Available: {list(_CORRECTION_REGISTRY)}")
    return _CORRECTION_REGISTRY[name]


# Auto-register defaults
from malthusjax.stats.tests import paired_t, sign_test, tost, wilcoxon

register_test("wilcoxon", wilcoxon)
register_test("paired_t", paired_t)
register_test("sign", sign_test)
register_test("tost", tost)

from malthusjax.stats.effects import cohens_dz, glass_delta, rank_biserial

register_effect("cohens_dz", cohens_dz)
register_effect("rank_biserial", rank_biserial)
register_effect("glass_delta", glass_delta)

from malthusjax.stats.correction import fdr_bh, holm_bonferroni

register_correction("holm", holm_bonferroni)
register_correction("fdr_bh", fdr_bh)

from malthusjax.stats.diagnostics import breusch_pagan, shapiro_wilk

register_diagnostic("breusch_pagan", breusch_pagan)
register_diagnostic("shapiro_wilk", shapiro_wilk)
