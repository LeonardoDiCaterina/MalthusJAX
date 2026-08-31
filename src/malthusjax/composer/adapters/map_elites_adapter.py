import time
from typing import Any


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
        import jax

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
            if hasattr(self.engine.emitter, "genome_config"):
                init_pop = self.engine.emitter.genome_config.init_population(k_init, self.pop_size)
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
            # Wrap the shared array into the correct Population PyTree
            init_pop_copy = jax.numpy.array(init_pop, copy=True)
            if hasattr(self.engine.emitter, "genome_config"):
                dummy_pop = self.engine.emitter.genome_config.init_population(k_init, self.pop_size)
                if hasattr(dummy_pop.genes, "replace"):
                    new_genes = dummy_pop.genes.replace(values=init_pop_copy)
                    init_pop = dummy_pop.replace(genes=new_genes)
                else:
                    init_pop = dummy_pop.replace(genes=init_pop_copy)
            else:
                # If we passed an ndarray for TensorNEAT, it is not well supported, so fail gracefully
                raise NotImplementedError(
                    "Passing ndarray init_pop to TensorNeatEmitter is not supported yet."
                )
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

        t_exec_start = time.perf_counter()
        final_state, scan_history, _ = self.engine.run(state, time_it=True, compile=True)
        t_exec_end = time.perf_counter()

        num_gens = int(self.engine.engine_params.num_generations)
        history = []
        track_keys = self.history_metrics or ["best_fitness", "qd_score", "coverage"]

        sign = -1.0 if self.maximize else 1.0

        for g in range(num_gens):
            gen_stats = {"generation": g + 1}
            for k in track_keys:
                if hasattr(scan_history, k):
                    val = getattr(scan_history, k)[g]
                    if k in ("best_fitness", "mean_fitness", "std_fitness"):
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

        summary = {
            "best_fitness": float(final_state.best_fitness * sign),
            "qd_score": float(qd_score) if qd_score is not None else 0.0,
            "coverage": float(coverage) if coverage is not None else 0.0,
            "final_generation": int(final_state.generation),
            "total_evaluations": int(final_state.generation * self.pop_size),
        }

        return {
            "history": history,
            "summary": summary,
            "timings": {"total": t_exec_end - t_exec_start},
        }
