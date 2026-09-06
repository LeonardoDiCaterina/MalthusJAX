"""Dynamic TOML parser for composable evaluators.

This module provides utilities to instantiate Evaluators, Environments,
Interpreters, Outputs, and Transforms directly from dictionary configurations,
typically loaded via TOML.
"""

from typing import Any, Dict, Optional

from malthusjax.core.fitness.composable import (
    base,
    environments,
    evaluators,
    interpreters,
)

try:
    import tensorneat
    TENSORNEAT_AVAILABLE = True
except ImportError:
    TENSORNEAT_AVAILABLE = False


def _instantiate_from_module(module: Any, class_name: str, kwargs: Dict[str, Any]) -> Any:
    """Helper to instantiate a class from a given module by name."""
    if not hasattr(module, class_name):
        raise ValueError(f"Class '{class_name}' not found in module '{module.__name__}'.")

    cls = getattr(module, class_name)
    # If the class defines a `create` classmethod, prefer it over `__init__`.
    if hasattr(cls, "create") and callable(cls.create):
        return cls.create(**kwargs)
    return cls(**kwargs)


def _parse_component(config: Any, module: Any, default_class: Optional[str] = None) -> Any:
    """Parses a component configuration (dict or string) into an instantiated object.
    
    If config is a string, it's assumed to be the class name with no arguments.
    If it's a dict, it must have a 'type' key (or fallback to default_class).
    """
    if config is None:
        return None

    if isinstance(config, str):
        class_name = config
        kwargs = {}
    elif isinstance(config, dict):
        kwargs = config.copy()
        class_name = kwargs.pop("type", default_class)
        if not class_name:
            raise ValueError(f"Missing 'type' key in configuration: {config}")
    else:
        # If it's already instantiated, return it.
        return config

    return _instantiate_from_module(module, class_name, kwargs)


def parse_evaluator(config: Dict[str, Any]) -> evaluators.BaseComposableEvaluator[Any, Any, Any, Any]:
    """Parses a nested configuration dictionary into a Composable Evaluator.
    
    Example config:
    {
        "type": "OptimizationEvaluator",
        "env": {"type": "BraxEnv", "env_name": "ant"},
        "interpreter": {"type": "MLPInterpreter", "hidden": [32, 32]},
        "output": {"type": "ScalarOutput", "maximize": True}
    }
    """
    if not isinstance(config, dict):
        raise TypeError(f"parse_evaluator expects a dictionary, got {type(config)}")

    kwargs = config.copy()
    evaluator_type = kwargs.pop("type", "OptimizationEvaluator")

    # 1. Parse Environment
    if "env" in kwargs:
        if isinstance(kwargs["env"], dict) and kwargs["env"].get("type") == "TensorNEATProblemWrapper":
            # Special case for TensorNEAT wrappers which need a problem instance
            if not TENSORNEAT_AVAILABLE:
                raise ImportError("TensorNEAT is not available, but TensorNEATProblemWrapper was requested.")
            prob_dict = kwargs["env"].copy()
            prob_dict.pop("type", None)
            prob_type = prob_dict.pop("problem_type", "XOR")
            import tensorneat.problem
            try:
                from tensorneat.problem.func_fit import xor
                # simple mapping for now
                if prob_type == "XOR":
                    problem = xor.XOR()
                else:
                    problem = getattr(tensorneat.problem, prob_type)(**prob_dict)
            except AttributeError:
                 problem = getattr(tensorneat.problem, prob_type)(**prob_dict)

            kwargs["env"] = environments.TensorNEATProblemWrapper(problem=problem)
        else:
            kwargs["env"] = _parse_component(kwargs["env"], environments)

    # 2. Parse Interpreter
    if "interpreter" not in kwargs:
        kwargs["interpreter"] = {"type": "IdentityInterpreter"}

    if isinstance(kwargs["interpreter"], dict) and kwargs.get("env"):
        env_inst = kwargs["env"]
        if hasattr(env_inst, "obs_dim") and "input_dim" not in kwargs["interpreter"]:
            kwargs["interpreter"]["input_dim"] = env_inst.obs_dim
        if hasattr(env_inst, "action_dim") and "output_dim" not in kwargs["interpreter"]:
            kwargs["interpreter"]["output_dim"] = env_inst.action_dim
    kwargs["interpreter"] = _parse_component(kwargs["interpreter"], interpreters, default_class="IdentityInterpreter")

    # 3. Parse Output
    if "output" not in kwargs:
        kwargs["output"] = {"type": "ScalarOutput"}
    kwargs["output"] = _parse_component(kwargs["output"], base, default_class="ScalarOutput")

    # 4. Parse Transform
    if "transform" not in kwargs:
        kwargs["transform"] = {"type": "IdentityTransform"}
    kwargs["transform"] = _parse_component(kwargs["transform"], base, default_class="IdentityTransform")

    # Instantiate the Evaluator itself
    return _instantiate_from_module(evaluators, evaluator_type, kwargs)
