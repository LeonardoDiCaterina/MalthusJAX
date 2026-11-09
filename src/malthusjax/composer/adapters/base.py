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
from malthusjax.core.logger import (
    StepLoggingConfig,
    _host_log_nan_anomaly,
    _host_log_step,
    get_logger,
)
from malthusjax.core.timing import TimingCapture, block_until_ready, execution_metadata

logger = get_logger("composer.adapters")


def _has_block_until_ready(obj: Any) -> bool:
    try:
        return callable(getattr(obj, "block_until_ready", None))
    except Exception:
        return False


def _block_all_until_ready(obj: Any) -> None:
    """Compatibility wrapper around the shared, error-propagating PyTree barrier."""
    block_until_ready(obj, effects=True)


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
        metrics_catalog: list[Any],
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
        step_logging: Optional[StepLoggingConfig] = None,
    ) -> None:
        self.framework_obj = framework_obj
        self.framework_params = framework_params
        self.init_fn = init_fn
        self.step_fn = step_fn
        self.eval_mode = eval_mode
        self.eval_translator = eval_translator
        self.metrics_catalog = metrics_catalog
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
        self.step_logging = step_logging

        self._jit_run_loop = None
        self._jit_run_spec = None

    def _build_jit_loop(self) -> Any:
        """Build and cache the JIT-compiled evolution loop."""
        # Replaced static objects/options require preparation of a new loop.
        # In-place mutation of third-party strategy internals is unsupported;
        # callers must construct a new adapter for those changes.
        spec = (
            id(self.framework_obj),
            id(self.framework_params),
            id(self.step_fn),
            id(self.eval_translator),
            id(self.evaluator),
            self.num_generations,
            self.maximize,
            self.backend_maximizes,
            self.step_logging,
            tuple(
                (m.name, id(m.source) if callable(m.source) else m.source, m.is_objective_value)
                for m in self.metrics_catalog
            ),
        )
        if self._jit_run_loop is not None and self._jit_run_spec == spec:
            return self._jit_run_loop

        def wrapped_eval_translator(*args: Any, **kwargs: Any) -> Any:
            return self.eval_translator(*args, **kwargs)

        step_logging = self.step_logging
        active_step_logging = step_logging is not None and step_logging.is_active()

        def scan_step(carry: Tuple[Any, Any], step_idx: Any) -> Tuple[Tuple[Any, Any], Any]:
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
            for metric in self.metrics_catalog:
                if callable(metric.source):
                    val = metric.source(metrics)
                else:
                    val = metrics.get(metric.source, jnp.nan)

                if self.backend_maximizes != self.maximize and metric.is_objective_value:
                    val = -val

                normalized_metrics[metric.name] = val

            if active_step_logging and step_logging is not None:
                gen = jnp.int32(1) + step_idx
                best_fit = normalized_metrics.get("best_fitness", jnp.nan)
                mean_fit = normalized_metrics.get("mean_fitness", jnp.nan)

                def _log_step_cb():
                    jax.debug.callback(
                        _host_log_step,
                        gen,
                        best_fit,
                        mean_fit,
                        step_logging.logger_name,
                    )

                if step_logging.log_interval is not None and step_logging.log_interval > 0:
                    jax.lax.cond(
                        (gen % step_logging.log_interval) == 0,
                        _log_step_cb,
                        lambda: None,
                    )

                if step_logging.log_nan_watchdog:

                    def _nan_watchdog_cb():
                        jax.debug.callback(
                            _host_log_nan_anomaly,
                            gen,
                            best_fit,
                            "best_fitness",
                            step_logging.logger_name,
                        )

                    jax.lax.cond(
                        jnp.isnan(best_fit) | jnp.isinf(best_fit),
                        _nan_watchdog_cb,
                        lambda: None,
                    )

            return (rng, state), normalized_metrics

        def run_loop(rng: Any, state_init: Any) -> Tuple[Any, Any]:
            carry = (rng, state_init)
            scan_xs = jnp.arange(self.num_generations) if active_step_logging else None
            carry, metrics = jax.lax.scan(
                scan_step, carry, scan_xs, length=self.num_generations, unroll=1
            )
            return carry[1], metrics

        self._jit_run_loop = jax.jit(run_loop)  # type: ignore[assignment]
        self._jit_run_spec = spec
        return self._jit_run_loop

    def _build_python_loop(self) -> Any:
        """Build a python loop for non-jittable frameworks."""

        def wrapped_eval_translator(*args: Any, **kwargs: Any) -> Any:
            return self.eval_translator(*args, **kwargs)

        step_logging = self.step_logging
        active_step_logging = step_logging is not None and step_logging.is_active()

        def scan_step(carry: Tuple[Any, Any], step_idx: Any) -> Tuple[Tuple[Any, Any], Any]:
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
            for metric in self.metrics_catalog:
                if callable(metric.source):
                    val = metric.source(metrics)
                else:
                    val = metrics.get(metric.source, jnp.nan)

                if self.backend_maximizes != self.maximize and metric.is_objective_value:
                    val = -val

                normalized_metrics[metric.name] = val

            return (rng, state), normalized_metrics

        def run_loop(rng: Any, state_init: Any) -> Tuple[Any, Any]:
            carry = (rng, state_init)
            metrics_history = []

            for g in range(self.num_generations):
                carry, metrics = scan_step(carry, g)
                metrics_history.append(metrics)
                if (
                    active_step_logging
                    and step_logging is not None
                    and step_logging.log_interval is not None
                    and ((g + 1) % step_logging.log_interval == 0)
                ):
                    best_fit = metrics.get("best_fitness", float("nan"))
                    mean_fit = metrics.get("mean_fitness", float("nan"))
                    _host_log_step(g + 1, best_fit, mean_fit, step_logging.logger_name)

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
        logger.debug(
            "UniversalAdapterEngine starting run for %d generations (pop_size=%d)...",
            self.num_generations,
            self.pop_size,
        )
        t_warmup_start = time.perf_counter()
        capture = TimingCapture()

        key, key_init, key_eval = jax.random.split(key, 3)

        # Initialize framework state
        state_init = self.init_fn(
            self.framework_obj, key_init, self.framework_params, self.initial_population
        )
        block_until_ready((key, state_init), effects=True)

        capture.initialized((key, state_init), self)
        # Build JIT loop or Python loop
        if self.use_python_loop:
            run_loop = self._build_python_loop()
        else:
            run_loop = self._build_jit_loop()
            if compile:
                run_loop = run_loop.lower(key, state_init).compile()

        capture.prepared()
        t_warmup_end = time.perf_counter()

        # Execute
        t_exec_start = time.perf_counter()
        final_state, scan_history = run_loop(key, state_init)

        # Block until all GPU computations are finished
        _block_all_until_ready((final_state, scan_history))

        t_exec_end = time.perf_counter()
        capture.executed()

        # Format history output
        history = []

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
                    gen_stats[k] = float(val)
                    if k == "best_fitness":
                        fitness_auc += float(val)
            history.append(gen_stats)

        report_best = (
            float(scan_history["best_fitness"][-1]) if "best_fitness" in scan_history else 0.0
        )

        summary = {
            "best_fitness": report_best,
            "fitness_auc": fitness_auc,
            "final_generation": self.num_generations,
            "total_evaluations": self.num_generations * self.pop_size,
        }

        # Inject any other tracked metrics
        for k in track_keys:
            if k in scan_history:
                val = scan_history[k][-1]
                summary[k] = float(val)

        mjx_evaluator = getattr(self, "malthusjax_evaluator", None) or self.evaluator
        if mjx_evaluator is not None and hasattr(mjx_evaluator, "get_gap_to_optimum"):
            gap = mjx_evaluator.get_gap_to_optimum(report_best)
            if gap is not None:
                summary["gap_to_optimum"] = float(gap)

        timings = {
            "warmup": t_warmup_end - t_warmup_start,
            "total": t_exec_end - t_warmup_start,
        }
        unavailable = None
        if self.use_python_loop:
            unavailable = "Python-loop mode does not provide prepared compiled execution timing"
        elif not compile:
            unavailable = "Compilation was not prepared; execution may include tracing/compilation"
        else:
            timings["execution"] = t_exec_end - t_exec_start

        logger.debug(
            "UniversalAdapterEngine completed in %.4fs (warmup: %.4fs, exec: %s)",
            timings["total"],
            timings["warmup"],
            timings.get("execution", "unavailable"),
        )

        return {
            "history": history,
            "summary": summary,
            "timings": timings,
            "timing_metadata": capture.metadata(unavailable),
            **({"metric_metadata": self.metric_metadata} if hasattr(self, "metric_metadata") else {}),
        }

    def get_supported_metrics(self) -> list[Any]:
        """Returns the catalog of all metrics supported by this adapter."""
        return self.metrics_catalog
