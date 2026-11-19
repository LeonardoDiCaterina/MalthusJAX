import time
from typing import Any

import numpy as np

from malthusjax.composer.adapters.metrics import archive_metric_metadata
from malthusjax.composer.design_exports import encoded_solution, selected_genotypes
from malthusjax.composer.engine_protocol import EngineOutputOptions
from malthusjax.composer.initial_population import prepare_initial_population
from malthusjax.core.timing import TimingCapture, block_until_ready


class MapElitesEngineAdapter:
    """Adapter to make Native MapElitesEngine compatible with BenchmarkRunner."""

    def __init__(
        self, engine, pop_size, maximize, history_metrics, initial_population=None, centroids=None
    ):
        self.engine = engine
        self.pop_size = pop_size
        self.maximize = maximize
        self.history_metrics = history_metrics
        self.initial_population = initial_population
        self.centroids = centroids

    def run_once(self, key):
        return self.run_with_options(key, EngineOutputOptions())

    def run_with_options(self, key, options: EngineOutputOptions):
        import jax

        t_warmup_start = time.perf_counter()
        capture = TimingCapture()

        # Check if we are running the exact QDAX replica for bit-parity
        is_qdax_replica = (
            getattr(self.engine, "engine_params", None)
            and getattr(self.engine.engine_params, "key_derivation", None) == "qdax_replica"
        )

        if is_qdax_replica:
            # Match UniversalAdapterEngine.run_once exact key flow:
            # key, key_init, key_eval = split(key, 3)
            _, k_init_qdax, _ = jax.random.split(key, 3)
            # The QDAX adapter stores key_init directly as its randkey in state.
            # But MapElitesEngine.init_state does k1, k2 = split(rng_key).
            # We want rng_key inside the state to be EXACTLY k_init_qdax.
            # We'll pass a dummy key and overwrite it after.
            k_init, k_run = jax.random.split(key)
        else:
            k_init, k_run = jax.random.split(key)

        init_pop = self.initial_population
        if init_pop is None:
            genome_config = getattr(self.engine.emitter, "genome_config", None)
            if genome_config is not None:
                init_pop = genome_config.init_population(k_init, self.pop_size)
            elif (
                hasattr(self.engine.emitter, "genome")
                and "TensorNeat" in self.engine.emitter.__class__.__name__
            ):
                import jax.numpy as jnp

                from malthusjax.core.genome.tensorneat_genome import (
                    TensorNeatGenome,
                    TensorNeatPopulation,
                )

                try:
                    from tensorneat.common import State
                except ImportError:
                    State = Any
                tn_state = State(randkey=k_init, generation=jnp.float32(0))
                # TensorNEAT initialize creates a single genome. We vmap it to create a population.
                import jax

                pop_keys = jax.random.split(k_init, self.pop_size)
                nodes, conns = jax.vmap(self.engine.emitter.genome.initialize, in_axes=(None, 0))(
                    tn_state, pop_keys
                )
                init_pop = TensorNeatPopulation(
                    genes=TensorNeatGenome(values=(nodes, conns)),
                    fitness=jnp.full((self.pop_size,), -jnp.inf),
                    config=None,
                    info={},
                )
            else:
                raise AttributeError(
                    "Emitter lacks genome_config or genome to generate initial population."
                )
        elif isinstance(init_pop, jax.numpy.ndarray):
            genome_config = getattr(self.engine.emitter, "genome_config", None)
            if genome_config is None:
                raise ValueError("Array initialization requires an emitter genome configuration")
            init_pop = prepare_initial_population(init_pop, genome_config, self.pop_size)
        elif isinstance(init_pop, tuple) and len(init_pop) == 2:
            # Wrap the TensorNEAT tuple (nodes, conns) in a TensorNeatPopulation
            from malthusjax.core.genome.tensorneat_genome import (
                TensorNeatGenome,
                TensorNeatPopulation,
            )

            genes = TensorNeatGenome(values=init_pop)
            init_pop = TensorNeatPopulation(
                genes=genes, fitness=jax.numpy.zeros(self.pop_size), config=None
            )

        # Copy centroids to prevent "Buffer has been deleted or donated" error across seeds
        centroids_copy = (
            jax.numpy.array(self.centroids, copy=True) if self.centroids is not None else None
        )

        # We copy the population natively using the newly added `.copy()` to safely prevent JAX donation bugs across seeds
        state = self.engine.init_state(k_run, init_pop.copy(), centroids_copy)

        if is_qdax_replica:
            # Force the exact QDAX key into the state for generation 1
            state = state.replace(rng_key=k_init_qdax)

        capture.initialized(state, self.engine)
        prepared = self.engine.prepare_run(state, return_history=options.collect_history)
        capture.prepared()
        t_warmup_end = time.perf_counter()

        t_exec_start = time.perf_counter()
        final_state, scan_history, _ = self.engine.run(
            state,
            time_it=True,
            compile=True,
            prepared_run=prepared,
            return_history=options.collect_history,
        )
        block_until_ready((final_state, scan_history), effects=True)
        t_exec_end = time.perf_counter()
        capture.executed()

        num_gens = int(self.engine.engine_params.num_generations)
        history = []
        track_keys = self.history_metrics or [
            "best_fitness",
            "mean_fitness",
            "std_fitness",
            "qd_score",
            "coverage",
        ]

        sign = -1.0 if self.maximize else 1.0

        track_metrics = self.engine.engine_params.track_metrics
        unavailable_metrics = {}
        if not track_metrics:
            unavailable_metrics = {
                name: "Archive metric tracking disabled"
                for name in ("mean_fitness", "std_fitness", "qd_score", "coverage")
            }
        elif not options.collect_history:
            unavailable_metrics = {
                name: "Generation history was not collected"
                for name in ("mean_fitness", "std_fitness")
            }

        for g in range(num_gens if options.collect_history else 0):
            gen_stats = {"generation": g + 1}
            for k in track_keys:
                if k not in unavailable_metrics and hasattr(scan_history, k):
                    val = getattr(scan_history, k)[g]
                    if k in ("best_fitness", "mean_fitness"):
                        val = val * sign
                    gen_stats[k] = float(val)
            history.append(gen_stats)

        # Safely extract qd_score and coverage. If they exist on final_state use them,
        # otherwise try to get the last element from scan_history.
        qd_score = getattr(final_state, "qd_score", None)
        if qd_score is None and hasattr(scan_history, "qd_score"):
            qd_score = scan_history.qd_score[-1]

        coverage = getattr(final_state, "coverage", None)
        if coverage is None and hasattr(scan_history, "coverage"):
            coverage = scan_history.coverage[-1]

        best_fitness = final_state.best_fitness
        if not options.collect_history and track_metrics:
            from malthusjax.core.fitness.qd.metrics import archive_metrics

            terminal = archive_metrics(final_state.repertoire.fitnesses)
            best_fitness = -terminal["max_fitness"]
            qd_score, coverage = terminal["qd_score"], terminal["coverage"]

        summary = {
            "best_fitness": float(best_fitness * sign),
            "final_generation": int(final_state.generation),
            "total_evaluations": int(final_state.generation * self.pop_size),
        }

        if track_metrics:
            summary["qd_score"] = float(qd_score) if qd_score is not None else float("nan")
            summary["coverage"] = float(coverage) if coverage is not None else float("nan")
        if history and track_metrics:
            for name in ("mean_fitness", "std_fitness"):
                summary[name] = history[-1].get(name, float("nan"))

        designs = {}
        if options.capture_solution or options.export_archive:
            repertoire = final_state.repertoire
            fitnesses = np.asarray(jax.device_get(repertoire.fitnesses)).reshape(-1)
            occupied = np.flatnonzero(np.isfinite(fitnesses))
            if options.capture_solution and occupied.size:
                index = int(occupied[np.argmax(fitnesses[occupied])])
                genome = jax.tree.map(lambda leaf: leaf[index], repertoire.genotypes)
                designs["solution"] = encoded_solution(
                    genome, float(-sign * fitnesses[index]), "archive_best"
                )
            if options.export_archive:
                descriptors, centroids = jax.device_get(
                    (repertoire.descriptors, repertoire.centroids)
                )
                designs["archive"] = {
                    "cell_index": occupied,
                    "archive_fitness": fitnesses[occupied],
                    "descriptors": descriptors[occupied],
                    "centroids": centroids,
                    **selected_genotypes(repertoire.genotypes, occupied),
                }

        return {
            **designs,
            "history": history,
            "summary": summary,
            "unavailable_metrics": unavailable_metrics,
            "timing_metadata": capture.metadata(),
            "metric_metadata": archive_metric_metadata(self.maximize),
            "timings": {
                "warmup": t_warmup_end - t_warmup_start,
                "execution": t_exec_end - t_exec_start,
                "total": t_exec_end - t_warmup_start,
            },
        }
