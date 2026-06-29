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


class UniversalAdapterEngine:
    """Universal adapter to make external frameworks compatible with the BenchmarkRunner.Engine protocol.
    
    Implements the `run_once(key)` contract interchangeably with `GeneticEngineAdapter`.
    """

    def __init__(
        self,
        framework_obj: Any,
        framework_params: Any,
        init_fn: Callable,
        step_fn: Callable,
        eval_mode: str,
        eval_translator: Callable,
        metrics_mapping: Dict[str, str | Callable],
        pop_size: int,
        num_generations: int,
        maximize: bool = False,
        initial_population: chex.Array = None,
        evaluator: Optional[BaseEvaluator[Any, Any, Any]] = None,
        history_metrics: Optional[Sequence[str]] = None,
        state_has_randkey: bool = False,
        use_python_loop: bool = False,
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
        self.history_metrics = history_metrics
        self.state_has_randkey = state_has_randkey
        self.use_python_loop = use_python_loop
        
        self._jit_run_loop = None

    def _build_jit_loop(self):
        """Build and cache the JIT-compiled evolution loop."""
        if self._jit_run_loop is not None:
            return self._jit_run_loop

        def scan_step(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
            rng, state = carry
            rng, key_step = jax.random.split(rng)
            
            # Step the framework
            if self.state_has_randkey:
                state, metrics = self.step_fn(self.framework_obj, state, key_step, self.framework_params, self.evaluator, self.eval_translator)
            else:
                state, metrics = self.step_fn(self.framework_obj, state, key_step, self.framework_params, self.evaluator, self.eval_translator)
                
            # Process metrics
            normalized_metrics = {}
            for k, v in self.metrics_mapping.items():
                if callable(v):
                    normalized_metrics[k] = v(metrics)
                else:
                    normalized_metrics[k] = metrics.get(v, jnp.nan)
                    
            return (rng, state), normalized_metrics

        def run_loop(rng: Any, state_init: Any) -> Tuple[Any, Any]:
            carry = (rng, state_init)
            carry, metrics = jax.lax.scan(scan_step, carry, None, length=self.num_generations, unroll=1)
            return carry[1], metrics

        self._jit_run_loop = jax.jit(run_loop)
        return self._jit_run_loop
        
    def _build_python_loop(self):
        """Build a python loop for non-jittable frameworks."""
        def scan_step(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Any]:
            rng, state = carry
            rng, key_step = jax.random.split(rng)
            
            # Step the framework
            state, metrics = self.step_fn(self.framework_obj, state, key_step, self.framework_params, self.evaluator, self.eval_translator)
                
            # Process metrics
            normalized_metrics = {}
            for k, v in self.metrics_mapping.items():
                if callable(v):
                    normalized_metrics[k] = v(metrics)
                else:
                    normalized_metrics[k] = metrics.get(v, jnp.nan)
                    
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

    def run_once(self, key: chex.Array, unroll_factor: int = 1, compile: bool = True) -> Dict[str, Any]:
        """Run one evolutionary experiment and return BenchmarkRunner-compatible results."""
        t_warmup_start = time.perf_counter()

        key, key_init, key_eval = jax.random.split(key, 3)

        # Initialize framework state
        state_init = self.init_fn(self.framework_obj, key_init, self.framework_params, self.initial_population)

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
        t_exec_end = time.perf_counter()

        # Format history output
        history = []
        sign = -1.0 if self.maximize else 1.0
        track_keys = self.history_metrics or ["best_fitness", "mean_fitness", "std_fitness"]

        fitness_auc = 0.0
        for g in range(self.num_generations):
            gen_stats: Dict[str, Any] = {"generation": g + 1}
            for k in track_keys:
                if k in scan_history:
                    val = scan_history[k][g]
                    if k in ("best_fitness", "mean_fitness", "std_fitness"):
                        val = val * sign
                    gen_stats[k] = float(val)
                    if k == "best_fitness":
                        fitness_auc += float(val)
            history.append(gen_stats)

        report_best = float(scan_history["best_fitness"][-1] * sign) if "best_fitness" in scan_history else 0.0

        summary = {
            "best_fitness": report_best,
            "fitness_auc": fitness_auc,
            "final_generation": self.num_generations,
            "total_evaluations": self.num_generations * self.pop_size,
        }

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
