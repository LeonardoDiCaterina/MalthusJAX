"""Universal `@adapter` decorator for wrapping external evolutionary frameworks."""

from typing import Any, Callable, Dict, Optional, Sequence

from malthusjax.composer.adapters.base import UniversalAdapterEngine


def adapter(
    framework: str,
    state_mapping: Dict[str, str],
    eval_translators: Dict[str, Callable[..., Any]],
    metrics_mapping: Dict[str, str | Callable[..., Any]],
) -> Callable[..., Any]:
    """Decorator to generate a UniversalAdapterEngine subclass dynamically.

    Args:
        framework: Name of the framework (e.g., "evosax", "qdax").
        state_mapping: Dict[str, Any] mapping "init" and "step" keys to the actual framework method names.
        eval_translators: Dict[str, Any] mapping EvalMode constants ("native", "malthusjax") to callables
            that handle fitness evaluation for that framework.
        metrics_mapping: Dict[str, Any] mapping metric names to keys or callables to extract from the
            framework's metrics object.
    """

    def decorator(cls: type) -> type:
        class AdaptedEngine(cls):  # type: ignore[misc]
            def __init__(
                self,
                strategy: Any,
                params: Any,
                pop_size: int,
                num_generations: int,
                problem: Optional[Any] = None,
                problem_state: Optional[Any] = None,
                maximize: bool = False,
                initial_population: Any = None,
                eval_mode: str = "native",
                evaluator: Optional[Any] = None,
                history_metrics: Optional[Sequence[str]] = None,
                use_python_loop: bool = False,
                **kwargs: Any,
            ) -> None:
                super().__init__()

                # Save extra args as attributes for the init/step hooks
                self.strategy = strategy
                self.params = params
                self.pop_size = pop_size
                self.num_generations = num_generations
                self.problem = problem
                self.problem_state = problem_state
                self.maximize = maximize
                self.initial_population = initial_population
                self.eval_mode = eval_mode
                self.evaluator = evaluator
                self.history_metrics = history_metrics
                self.use_python_loop = use_python_loop
                for k, v in kwargs.items():
                    setattr(self, k, v)

                # Retrieve bridging methods
                init_fn = getattr(self, "_adapter_init", None)
                if init_fn is None:
                    raise NotImplementedError(f"Class {cls.__name__} must implement _adapter_init")

                step_fn = getattr(self, "_adapter_step", None)
                if step_fn is None:
                    raise NotImplementedError(f"Class {cls.__name__} must implement _adapter_step")

                eval_translator = eval_translators.get(eval_mode)
                if eval_translator is None:
                    raise ValueError(f"EvalMode '{eval_mode}' is not supported by {cls.__name__}")

                # Bundle the problem for NATIVE evaluation
                framework_evaluator: Any
                if eval_mode == "native":
                    framework_evaluator = (problem, problem_state)
                else:
                    framework_evaluator = evaluator

                # Evosax adapter accesses this attribute during _adapter_init
                self._framework_evaluator = framework_evaluator

                self.engine = UniversalAdapterEngine(
                    framework_obj=strategy,
                    framework_params=params,
                    init_fn=init_fn,
                    step_fn=step_fn,
                    eval_mode=eval_mode,
                    eval_translator=eval_translator,
                    metrics_mapping=metrics_mapping,
                    pop_size=pop_size,
                    num_generations=num_generations,
                    maximize=maximize,
                    initial_population=initial_population,
                    evaluator=framework_evaluator,
                    malthusjax_evaluator=evaluator,
                    history_metrics=history_metrics,
                    state_has_randkey=False,
                    use_python_loop=use_python_loop,
                )

            def run_once(self, key: Any, unroll_factor: int = 1, compile: bool = True) -> Any:
                return self.engine.run_once(key, unroll_factor, compile)

        AdaptedEngine.__name__ = f"{cls.__name__}Adapted"
        return AdaptedEngine

    return decorator
