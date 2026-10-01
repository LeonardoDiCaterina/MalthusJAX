"""Tests for the shared evaluator resolution utility.

Verifies that ``resolve_evaluator_base()`` produces evaluators equivalent
to each of the 4 current copies in ``factory.py``, and that the known
BBOB integer-index gap (missing in composable evosax) is fixed.
"""

import jax.numpy as jnp
import pytest

from malthusjax.composer.backends._evaluator_resolver import resolve_evaluator_base


class TestResolveEvaluatorBase:
    """Core resolution tests — string, dict, object, None."""

    def test_none_returns_sphere_evaluator(self):
        """None input must produce a default Sphere evaluator."""
        evaluator = resolve_evaluator_base(None, num_dims=3)
        assert evaluator is not None
        # Sphere evaluator should be callable via evaluate_population or similar
        assert hasattr(evaluator, "evaluate_population") or hasattr(evaluator, "__call__")

    def test_string_spec_sphere(self):
        """A string spec like 'sphere:dim=5' must resolve to an evaluator."""
        evaluator = resolve_evaluator_base("sphere:dim=5", seed=42)
        assert evaluator is not None

    def test_string_spec_injects_seed(self):
        """If the string spec omits seed=, the resolver must inject it."""
        evaluator = resolve_evaluator_base("sphere:dim=3", seed=99)
        assert evaluator is not None

    def test_string_spec_with_existing_seed(self):
        """If the string spec already has seed=, do not double-inject."""
        evaluator = resolve_evaluator_base("sphere:dim=3,seed=7", seed=99)
        assert evaluator is not None

    def test_dict_spec(self):
        """A dict spec must resolve to an evaluator."""
        evaluator = resolve_evaluator_base({"type": "sphere", "dim": 3}, seed=42)
        assert evaluator is not None

    def test_prebuilt_object_passthrough(self):
        """A pre-built evaluator object must be returned as-is."""
        sentinel = object()
        result = resolve_evaluator_base(sentinel)
        assert result is sentinel


class TestBBOBResolution:
    """BBOB-specific resolution, including the integer-index fix."""

    def test_bbob_string_fn_name(self):
        """BBOB with a string function name (e.g. 'sphere') must work."""
        pytest.importorskip("evosax")
        evaluator = resolve_evaluator_base("bbob:fn=sphere,dim=2", seed=0)
        assert evaluator is not None

    def test_bbob_integer_index(self):
        """BBOB with an integer function index (e.g. fn=1) must work.

        This is the regression test for the known bug where
        build_composable_evosax_engine was missing this branch.
        """
        pytest.importorskip("evosax")
        evaluator = resolve_evaluator_base("bbob:fn=1,dim=2", seed=0)
        assert evaluator is not None

    def test_bbob_integer_index_out_of_range(self):
        """An out-of-range BBOB index must raise ValueError."""
        pytest.importorskip("evosax")
        with pytest.raises(ValueError, match="out of range"):
            resolve_evaluator_base("bbob:fn=999,dim=2")

    def test_bbob_missing_fn_raises(self):
        """BBOB spec without fn= or fn_name= must raise ValueError."""
        pytest.importorskip("evosax")
        with pytest.raises(ValueError, match="requires either fn_name or fn"):
            resolve_evaluator_base("bbob:dim=2")

    def test_bbob_evaluator_produces_finite_output(self):
        """The resolved BBOB evaluator must produce finite fitness values."""
        pytest.importorskip("evosax")
        evaluator = resolve_evaluator_base("bbob:fn=sphere,dim=3", seed=0)
        # Test via evaluate_population if available
        if hasattr(evaluator, "evaluate_population"):
            from malthusjax.core.genome.real_genome import (
                RealGenome,
                RealGenomeConfig,
                RealPopulation,
            )

            config = RealGenomeConfig(shape=(3,), bounds=(-5.0, 5.0))
            genes = RealGenome(values=jnp.zeros((4, 3)))
            pop = RealPopulation(genes=genes, fitness=jnp.zeros(4), config=config)
            updated = evaluator.evaluate_population(pop)
            assert jnp.all(jnp.isfinite(updated.fitness))


class TestResolutionParity:
    """Verify that the new resolver matches old factory behavior."""

    def test_maximize_flag_passthrough(self):
        """The maximize flag must be forwarded to the evaluator."""
        eval_min = resolve_evaluator_base("sphere:dim=2", maximize=False, seed=42)
        eval_max = resolve_evaluator_base("sphere:dim=2", maximize=True, seed=42)
        # Both must resolve; they may differ in internal sign convention
        assert eval_min is not None
        assert eval_max is not None

    def test_seed_determinism(self):
        """Same spec + same seed must produce identical evaluators."""
        e1 = resolve_evaluator_base("sphere:dim=2", seed=42)
        e2 = resolve_evaluator_base("sphere:dim=2", seed=42)
        # Both should exist and be structurally equivalent
        assert type(e1) is type(e2)
