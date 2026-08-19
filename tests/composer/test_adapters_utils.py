from unittest.mock import Mock

from malthusjax.composer.adapters.utils import resolve_bounds


def test_resolve_bounds_explicit():
    # Should prefer explicit bounds over everything else
    evaluator = Mock()
    evaluator.config.genome_config.bounds = (-1.0, 1.0)

    result = resolve_bounds(bounds=(-2.0, 2.0), evaluator=evaluator, caller_name="test")
    assert result == (-2.0, 2.0)

def test_resolve_bounds_evaluator_config():
    # Should use evaluator config if no explicit bounds provided
    evaluator = Mock()
    evaluator.config.genome_config.bounds = (-1.0, 1.0)

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test")
    assert result == (-1.0, 1.0)

def test_resolve_bounds_fallback(recwarn):
    # Should fallback and warn if neither explicit bounds nor evaluator config are provided
    evaluator = Mock()
    del evaluator.config # Remove config to trigger fallback

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test", default=(-5.0, 5.0))

    assert result == (-5.0, 5.0)
    assert len(recwarn) == 1
    warn_msg = str(recwarn[0].message)
    assert "No bounds were explicitly provided to `test`" in warn_msg
    assert "Falling back to the default bounds of (-5.0, 5.0)" in warn_msg

def test_resolve_bounds_no_evaluator(recwarn):
    # Should handle evaluator being None
    result = resolve_bounds(bounds=None, evaluator=None, caller_name="test", default=(-10.0, 10.0))

    assert result == (-10.0, 10.0)
    assert len(recwarn) == 1

def test_resolve_bounds_missing_genome_config(recwarn):
    # Case 1: Partial attribute chain — missing genome_config
    evaluator = Mock(spec=["config"])
    del evaluator.config.genome_config

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test")
    assert result == (-5.0, 5.0)
    assert len(recwarn) == 1

def test_resolve_bounds_missing_bounds(recwarn):
    # Case 2: Partial attribute chain — missing bounds
    evaluator = Mock()
    del evaluator.config.genome_config.bounds

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test")
    assert result == (-5.0, 5.0)
    assert len(recwarn) == 1

def test_resolve_bounds_bounds_is_none(recwarn):
    # Case 3: bounds attribute present but None
    evaluator = Mock()
    evaluator.config.genome_config.bounds = None

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test")
    assert result == (-5.0, 5.0)
    assert len(recwarn) == 1

def test_resolve_bounds_priority_ordering():
    # Case 5: Priority ordering (explicit > evaluator config)
    evaluator = Mock()
    evaluator.config.genome_config.bounds = (-1.0, 1.0)

    result = resolve_bounds(bounds=(-2.0, 2.0), evaluator=evaluator, caller_name="test")
    assert result == (-2.0, 2.0)

def test_resolve_bounds_malformed_evaluator_bounds(recwarn):
    # Case 6: Malformed bounds shape from evaluator
    evaluator = Mock()
    evaluator.config.genome_config.bounds = ("a", "b", "c") # Malformed 3-tuple of strings

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test")
    assert result == ("a", "b", "c")
    assert len(recwarn) == 0

def test_resolve_bounds_custom_default(recwarn):
    # Case 7: Custom default parameter
    evaluator = Mock()
    del evaluator.config

    result = resolve_bounds(bounds=None, evaluator=evaluator, caller_name="test", default=(-100.0, 100.0))
    assert result == (-100.0, 100.0)
    assert len(recwarn) == 1

def test_resolve_bounds_caller_name_interpolation(recwarn):
    # Case 8: caller_name interpolation at both real call sites
    evaluator = Mock()
    del evaluator.config

    # Evosax call site
    resolve_bounds(bounds=None, evaluator=evaluator, caller_name="build_evosax_engine")
    msg1 = str(recwarn[0].message)
    assert "No bounds were explicitly provided to `build_evosax_engine`" in msg1
    assert "bounds of (-5.0, 5.0) for evosax initialization" in msg1

    # QDAX call site
    resolve_bounds(bounds=None, evaluator=evaluator, caller_name="build_qdax_engine")
    msg2 = str(recwarn[1].message)
    assert "No bounds were explicitly provided to `build_qdax_engine`" in msg2
    assert "bounds of (-5.0, 5.0) for qdax initialization" in msg2

