"""Universal adapter core for integrating external JAX evolutionary frameworks.

This module provides the `UniversalAdapterEngine`, which abstracts the JAX
lax.scan compilation loop, timing methodology, and fitness evaluation mapping
so that any external framework (evosax, qdax, etc.) can be seamlessly dropped
into MalthusJAX's ecosystem.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

from malthusjax.core.fitness.base import BaseEvaluator


def _has_block_until_ready(obj: Any) -> bool:
    try:
        return callable(getattr(obj, "block_until_ready", None))
    except Exception:
        return False


def _block_all_until_ready(obj: Any) -> None:
    """Recursively block execution until all JAX arrays inside any object/struct/dict are ready."""
    if _has_block_until_ready(obj):
        obj.block_until_ready()
    elif isinstance(obj, dict):
        for v in obj.values():
            _block_all_until_ready(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _block_all_until_ready(item)
    else:
        try:
            d = vars(obj)
            for v in d.values():
                _block_all_until_ready(v)
        except Exception:
            pass


class UniversalAdapterEngine:
    """Universal adapter to make external frameworks compatible with the BenchmarkRunner.Engine protocol.

    Implements the `run_once(key)` contract interchangeably with `GeneticEngineAdapter`.
    """

    def __init__(
        self,
        framework_obj: Any,
        framework_params: Any,
        init_fn: Callable[..., Any],
        step_fn: Callable[..., Any],
        eval_mode: str,
        eval_translator: Callable[..., Any],
        metrics_mapping: Dict[str, str | Callable[..., Any]],
        pop_size: int,
        num_generations: int,
        maximize: bool = False,
        initial_population: chex.Array = None,
        evaluator: Optional[Any] = None,
        malthusjax_evaluator: Optional[BaseEvaluator[Any, Any, Any]] = None,
        history_metrics: Optional[Sequence[str]] = None,
        state_has_randkey: bool = False,
        use_python_loop: bool = False,
        backend_maximizes: bool = False,
    ) -> None:
        self.framework_obj = framework_obj
        self.framework_params = framework_params
        self.init_fn = init_fn
        self.step_fn = step_fn
        self.eval_mode = eval_mode
        self.eval_translator = eval_translator
        self.metrics_mapping = metrics_mapping
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.maximize = maximize
        self.initial_population = initial_population
        self.evaluator = evaluator
        self.malthusjax_evaluator = malthusjax_evaluator
        self.history_metrics = history_metrics
        self.state_has_randkey = state_has_randkey
        self.use_python_loop = use_python_loop
        self.backend_maximizes = backend_maximizes

        self._jit_run_loop = None

    def _build_jit_loop(self) -> Any:
        """Build and cache the JIT-compiled evolution loop."""
        if self._jit_run_loop is not None:
            return self._jit_run_loop

        def wrapped_eval_translator(evaluator: Any, pop: Any, state: Any, key: Any) -> Any:
            raw_fitness = self.eval_translator(evaluator, pop, state, key)
            return -raw_fitness if self.backend_maximizes else raw_fitness

        def scan_step(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
            rng, state = carry
            rng, key_step = jax.random.split(rng)

            # Step the framework
            if self.state_has_randkey:
                state, metrics = self.step_fn(
                    self.framework_obj,
                    state,
                    key_step,
                    self.framework_params,
                    self.evaluator,
                    wrapped_eval_translator,
                )
            else:
                state, metrics = self.step_fn(
                    self.framework_obj,
                    state,
                    key_step,
                    self.framework_params,
                    self.evaluator,
                    wrapped_eval_translator,
                )

            # Process metrics
            normalized_metrics = {}
            for k, v in self.metrics_mapping.items():
                if callable(v):
                    val = v(metrics)
                else:
                    val = metrics.get(v, jnp.nan)

                if self.backend_maximizes and k in ("best_fitness", "mean_fitness", "max_fitness", "fitness_auc", "qd_score"):
                    val = -val

                normalized_metrics[k] = val

            return (rng, state), normalized_metrics

        def run_loop(rng: Any, state_init: Any) -> Tuple[Any, Any]:
            carry = (rng, state_init)
            carry, metrics = jax.lax.scan(
                scan_step, carry, None, length=self.num_generations, unroll=1
            )
            return carry[1], metrics

        self._jit_run_loop = jax.jit(run_loop)  # type: ignore[assignment]
        return self._jit_run_loop

    def _build_python_loop(self) -> Any:
        """Build a python loop for non-jittable frameworks."""

        def wrapped_eval_translator(evaluator: Any, pop: Any, state: Any, key: Any) -> Any:
            raw_fitness = self.eval_translator(evaluator, pop, state, key)
            return -raw_fitness if self.backend_maximizes else raw_fitness

        def scan_step(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
            rng, state = carry
            rng, key_step = jax.random.split(rng)

            # Step the framework
            state, metrics = self.step_fn(
                self.framework_obj,
                state,
                key_step,
                self.framework_params,
                self.evaluator,
                wrapped_eval_translator,
            )

            # Process metrics
            normalized_metrics = {}
            for k, v in self.metrics_mapping.items():
                if callable(v):
                    val = v(metrics)
                else:
                    val = metrics.get(v, jnp.nan)

                if self.backend_maximizes and k in ("best_fitness", "mean_fitness", "max_fitness", "fitness_auc", "qd_score"):
                    val = -val

                normalized_metrics[k] = val

            return (rng, state), normalized_metrics

        def run_loop(rng: Any, state_init: Any) -> Tuple[Any, Any]:
            carry = (rng, state_init)
            metrics_history = []

            for _ in range(self.num_generations):
                carry, metrics = scan_step(carry, None)
                metrics_history.append(metrics)

            # Stack metrics to match lax.scan output format
            stacked_metrics = {}
            if len(metrics_history) > 0:
                for k in metrics_history[0].keys():
                    stacked_metrics[k] = jnp.stack([m[k] for m in metrics_history])

            return carry[1], stacked_metrics

        return run_loop

    def run_once(
        self, key: chex.Array, unroll_factor: int = 1, compile: bool = True
    ) -> Dict[str, Any]:
        """Run one evolutionary experiment and return BenchmarkRunner-compatible results."""
        t_warmup_start = time.perf_counter()

        key, key_init, key_eval = jax.random.split(key, 3)

        # Initialize framework state
        state_init = self.init_fn(
            self.framework_obj, key_init, self.framework_params, self.initial_population
        )

        # Build JIT loop or Python loop
        if self.use_python_loop:
            run_loop = self._build_python_loop()
        else:
            run_loop = self._build_jit_loop()
            if compile:
                _ = run_loop.lower(key, state_init).compile()

        t_warmup_end = time.perf_counter()

        # Execute
        t_exec_start = time.perf_counter()
        final_state, scan_history = run_loop(key, state_init)

        # Block until all GPU computations are finished
        _block_all_until_ready((final_state, scan_history))

        t_exec_end = time.perf_counter()

        # Format history output
        history = []
        sign = -1.0 if self.maximize else 1.0

        # If history_metrics is explicitly provided, filter by it, else include all available metrics
        if self.history_metrics:
            track_keys = self.history_metrics
            if "best_fitness" not in track_keys and "best_fitness" in scan_history:
                track_keys = list(track_keys) + ["best_fitness"]
        else:
            track_keys = list(scan_history.keys())

        fitness_auc = 0.0
        for g in range(self.num_generations):
            gen_stats: Dict[str, Any] = {"generation": g + 1}
            for k in track_keys:
                if k in scan_history:
                    val = scan_history[k][g]
                    if k in ("best_fitness", "mean_fitness"):
                        val = val * sign
                    gen_stats[k] = float(val)
                    if k == "best_fitness":
                        fitness_auc += float(val)
            history.append(gen_stats)

        report_best = (
            float(scan_history["best_fitness"][-1] * sign)
            if "best_fitness" in scan_history
            else 0.0
        )

        summary = {
            "best_fitness": report_best,
            "fitness_auc": fitness_auc,
            "final_generation": self.num_generations,
            "total_evaluations": self.num_generations * self.pop_size,
        }

        # Inject any other tracked metrics (e.g. qd_score, coverage) from their final generation
        for k in track_keys:
            if k not in ("best_fitness", "mean_fitness", "std_fitness"):
                if k in scan_history:
                    val = scan_history[k][-1]
                    summary[k] = float(val)
            elif k == "qd_score":
                if k in scan_history:
                    val = scan_history[k][-1]
                    summary[k] = float(val * sign)

        mjx_evaluator = getattr(self, "malthusjax_evaluator", None) or self.evaluator
        if mjx_evaluator is not None and hasattr(mjx_evaluator, "get_gap_to_optimum"):
            gap = mjx_evaluator.get_gap_to_optimum(report_best)
            if gap is not None:
                summary["gap_to_optimum"] = float(gap)

        timings = {
            "warmup": t_warmup_end - t_warmup_start,
            "execution": t_exec_end - t_exec_start,
            "total": t_exec_end - t_warmup_start,
        }

        return {
            "history": history,
            "summary": summary,
            "timings": timings,
        }
