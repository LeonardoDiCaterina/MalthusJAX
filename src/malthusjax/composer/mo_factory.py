"""Helpers for constructing MO engines for the Composer."""

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from malthusjax.core.timing import TimingCapture, block_until_ready, execution_metadata
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams


class MOEngineAdapter:
    """Adapter to make MOEngine compatible with BenchmarkRunner.Engine protocol."""

    def __init__(
        self,
        mo_engine: MOEngine[Any, Any],
        genome_config: Any,
        maximize: bool = True,
        initial_population: Any = None,
        history_metrics: Optional[Sequence[str]] = None,
    ):
        self.mo_engine = mo_engine
        self.genome_config = genome_config
        self.maximize = maximize
        self.initial_population = initial_population
        self.history_metrics = history_metrics

    def run_once(self, key: chex.Array) -> Dict[str, Any]:
        t_warmup_start = time.perf_counter()
        capture = TimingCapture()

        # Determine population initialization
        if self.initial_population is not None:
            arr = jnp.asarray(self.initial_population)
            pop = RealPopulation.from_array(arr, self.genome_config, RealGenome, axis=0)
        else:
            k_init, key = jax.random.split(key)
            pop = self.genome_config.init_population(k_init, self.mo_engine.engine_params.pop_size)

        state = self.mo_engine.init_state(key, pop)

        capture.initialized(state, self.mo_engine)
        prepared = self.mo_engine.prepare_run(state)
        capture.prepared()
        t_warmup_end = time.perf_counter()

        t_exec_start = time.perf_counter()
        final_state, scan_history, _ = self.mo_engine.run(
            state, time_it=True, compile=True, prepared_run=prepared
        )
        block_until_ready((final_state, scan_history), effects=True)
        t_exec_end = time.perf_counter()
        capture.executed()

        num_gens = int(self.mo_engine.engine_params.num_generations)
        history = []
        track_keys = self.history_metrics or [
            "num_pareto_optimal",
            "max_crowding_distance",
            "best_fitness",
            "mean_fitness",
            "std_fitness",
        ]

        for g in range(num_gens):
            gen_stats: Dict[str, Any] = {"generation": g + 1}
            for k in track_keys:
                if hasattr(scan_history, k):
                    val = getattr(scan_history, k)[g]
                    gen_stats[k] = float(val)
            history.append(gen_stats)

        summary = {
            "best_fitness": float(final_state.best_fitness),
            "final_generation": int(final_state.generation),
            "num_pareto_optimal": int(jnp.sum(final_state.population.pareto_rank == 0)),
            "total_evaluations": int(
                final_state.generation * self.mo_engine.engine_params.pop_size
            ),
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
            "timing_metadata": capture.metadata(),
        }


def build_mo_engine(
    fitness_evaluator: Any,
    emitter: Any,
    genome_type: str = "real",
    pop_size: int = 50,
    generations: int = 100,
    genome_shape: Tuple[int, ...] = (10,),
    bounds: Tuple[float, float] = (-5.0, 5.0),
    history_metrics: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> MOEngineAdapter:
    """Build a MOEngine from concrete operator instances."""

    if "genome_length" in kwargs:
        genome_shape = (int(kwargs.pop("genome_length")),)
    if isinstance(genome_shape, int):
        genome_shape = (genome_shape,)

    genome_config: Any
    if genome_type == "real":
        genome_config = RealGenomeConfig(
            shape=genome_shape, bounds=bounds, dtype=kwargs.get("dtype", "float32")
        )
    elif genome_type == "binary":
        genome_config = BinaryGenomeConfig(shape=genome_shape)
    else:
        raise ValueError(f"Unsupported genome type: {genome_type}")

    engine_params = MOEngineParams(
        pop_size=pop_size,
        num_generations=generations,
    )

    engine: Any = MOEngine(
        emitter=emitter, evaluator=fitness_evaluator, engine_params=engine_params
    )

    maximize_flag = getattr(fitness_evaluator.config, "maximize", False)

    initial_population = kwargs.get("initial_population", None)

    return MOEngineAdapter(
        engine,
        genome_config,
        maximize=maximize_flag,
        initial_population=initial_population,
        history_metrics=history_metrics,
    )
