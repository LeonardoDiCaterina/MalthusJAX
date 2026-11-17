"""Helpers for constructing MO engines for the Composer."""

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np

from malthusjax.composer.engine_protocol import EngineOutputOptions
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
from malthusjax.core.genome.categorical_genome import CategoricalGenomeConfig
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.core.timing import TimingCapture, block_until_ready
from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams

from .design_exports import selected_genotypes
from .initial_population import prepare_initial_population


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
        return self.run_with_options(key, EngineOutputOptions())

    def run_with_options(self, key: chex.Array, options: EngineOutputOptions) -> Dict[str, Any]:
        t_warmup_start = time.perf_counter()
        capture = TimingCapture()

        # Determine population initialization
        if self.initial_population is not None:
            pop = prepare_initial_population(
                self.initial_population, self.genome_config, self.mo_engine.engine_params.pop_size
            )
        else:
            k_init, key = jax.random.split(key)
            pop = self.genome_config.init_population(k_init, self.mo_engine.engine_params.pop_size)

        state = self.mo_engine.init_state(key, pop)

        capture.initialized(state, self.mo_engine)
        prepared = self.mo_engine.prepare_run(state, return_history=options.collect_history)
        capture.prepared()
        t_warmup_end = time.perf_counter()

        t_exec_start = time.perf_counter()
        final_state, scan_history, _ = self.mo_engine.run(
            state,
            time_it=True,
            compile=True,
            prepared_run=prepared,
            return_history=options.collect_history,
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

        for g in range(num_gens if options.collect_history else 0):
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

        designs = {}
        if options.export_pareto:
            population = final_state.population
            ranks, objectives = jax.device_get((population.pareto_rank, population.fitness))
            selected = np.flatnonzero((ranks == 0) & np.all(np.isfinite(objectives), axis=1))
            designs["pareto"] = {
                "population_index": selected,
                "objectives": objectives[selected],
                "objective_direction": np.full(
                    objectives.shape[1], "maximize" if population.maximize else "minimize"
                ),
                **selected_genotypes(population.genes, selected),
            }
        return {
            **designs,
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
    *,
    genome_config: Any = None,
    **kwargs: Any,
) -> MOEngineAdapter:
    """Build a MOEngine from concrete operator instances."""

    if "genome_length" in kwargs:
        genome_shape = (int(kwargs.pop("genome_length")),)
    if isinstance(genome_shape, int):
        genome_shape = (genome_shape,)

    explicit_config = genome_config is not None or genome_type == "categorical"
    if genome_config is None:
        if genome_type == "real":
            genome_config = RealGenomeConfig(
                shape=genome_shape, bounds=bounds, dtype=kwargs.get("dtype", "float32")
            )
        elif genome_type == "binary":
            genome_config = BinaryGenomeConfig(shape=genome_shape)
        elif genome_type == "categorical":
            genome_config = getattr(emitter, "genome_config", None)
            if not isinstance(genome_config, CategoricalGenomeConfig):
                raise ValueError("Categorical MO requires a CategoricalGenomeConfig")
        else:
            raise ValueError(f"Unsupported genome type: {genome_type}")

    emitter_config = getattr(emitter, "genome_config", None)
    if explicit_config and emitter_config is not None and emitter_config != genome_config:
        raise ValueError("MO emitter genome configuration does not match the engine")

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
