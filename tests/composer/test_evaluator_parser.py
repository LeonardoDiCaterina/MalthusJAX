"""Targeted coverage tests for malthusjax.composer.evaluator_parser."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from malthusjax.composer.evaluator_parser import (
    _parse_component,
    parse_evaluator,
)
from malthusjax.core.fitness.composable import interpreters


def test_parse_component_variations():
    # 1. config is None
    assert _parse_component(None, interpreters) is None

    # 2. config is a string (class name)
    interp = _parse_component("IdentityInterpreter", interpreters)
    assert isinstance(interp, interpreters.IdentityInterpreter)

    # 3. config is a dict with type
    interp_dict = _parse_component({"type": "IdentityInterpreter"}, interpreters)
    assert isinstance(interp_dict, interpreters.IdentityInterpreter)

    # 4. config is a dict using default_class
    interp_default = _parse_component({}, interpreters, default_class="IdentityInterpreter")
    assert isinstance(interp_default, interpreters.IdentityInterpreter)

    # 5. config is a dict missing type and no default_class -> raises ValueError
    with pytest.raises(ValueError, match="Missing 'type' key"):
        _parse_component({}, interpreters)

    # 6. config is already an instantiated object
    already = interpreters.IdentityInterpreter()
    assert _parse_component(already, interpreters) is already

    # 7. class does not exist in module -> raises ValueError
    with pytest.raises(ValueError, match="Class 'NonExistentClass' not found"):
        _parse_component("NonExistentClass", interpreters)


def test_parse_evaluator_type_error():
    with pytest.raises(TypeError, match="parse_evaluator expects a dictionary"):
        parse_evaluator(["not", "a", "dict"])


def test_parse_evaluator_bbob():
    config = {
        "type": "OptimizationEvaluator",
        "env": {"type": "BBOBEnv", "fn_name": "sphere", "num_dims": 2},
        "interpreter": "IdentityInterpreter",
        "output": {"type": "ScalarOutput", "maximize": True},
        "transform": "IdentityTransform",
    }
    evalr = parse_evaluator(config)
    assert evalr is not None
    assert hasattr(evalr, "evaluate")


def test_parse_evaluator_env_dimension_propagation():
    # Mock env with obs_dim and action_dim
    mock_env = SimpleNamespace(obs_dim=8, action_dim=2)
    config = {
        "type": "OptimizationEvaluator",
        "env": mock_env,
        "interpreter": {"type": "MLPInterpreter", "hidden": [16]},
        "output": "ScalarOutput",
    }
    evalr = parse_evaluator(config)
    assert evalr.env is mock_env
    assert evalr.interpreter.input_dim == 8
    assert evalr.interpreter.output_dim == 2


def test_parse_evaluator_tensorneat_problem_wrapper():
    config = {
        "type": "OptimizationEvaluator",
        "env": {
            "type": "TensorNEATProblemWrapper",
            "problem_type": "XOR",
        },
        "interpreter": "IdentityInterpreter",
        "output": "ScalarOutput",
    }
    evalr = parse_evaluator(config)
    assert evalr is not None

    # Test missing tensorneat import
    with patch("malthusjax.composer.evaluator_parser.TENSORNEAT_AVAILABLE", False):
        with pytest.raises(ImportError, match="TensorNEAT is not available"):
            parse_evaluator(config)
