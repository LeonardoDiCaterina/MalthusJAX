#!/usr/bin/env python3
"""Hopper NSGA-II/NEAT islands, composed using existing public components.

Script-local adapters expose NEAT parent genes and implement topology routing;
NSGA-II tournament, merge/survival, NEAT mutation and Hopper rollout are reused.
--smoke forces CPU and tiny single/ring runs. No cluster submission occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "packages/malthus_neat/src")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="CPU: P4, G2, horizon2, two islands, one migration")
    p.add_argument("--output", required=True, type=Path, help="Fresh output directory")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--topologies", nargs="+", choices=["single", "ring", "bidirectional", "fully_connected"], default=None)
    p.add_argument("--population", type=int, default=128, help="Total population, not per-island")
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--migration-interval", type=int, default=20)
    p.add_argument("--migrants-per-island", type=int, help="Total outgoing individuals per island/event, independent of degree")
    p.add_argument("--heldout-policy", choices=["all"], help="Explicit full-population evaluation; sampled policy remains unfrozen")
    p.add_argument("--max-nodes", type=int, default=32)
    p.add_argument("--max-conns", type=int, default=64)
    p.add_argument("--heldout-episodes", type=int, default=20)
    args = p.parse_args()
    if args.seeds is None:
        if not args.smoke:
            p.error("non-smoke requires an explicit --seeds list")
        args.seeds = [17]
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error("--seeds requires distinct uint32 seeds")
    if args.smoke:
        if len(args.seeds) > 2:
            p.error("smoke allows at most two seeds")
        args.population, args.islands, args.generations, args.horizon = 4, 2, 2, 2
        args.migration_interval, args.migrants_per_island, args.heldout_episodes = 1, 1, 1
        args.heldout_policy = "all"
        args.max_nodes, args.max_conns = 16, 40
        args.topologies = args.topologies or ["single", "ring"]
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        args.topologies = args.topologies or ["single", "ring", "bidirectional", "fully_connected"]
        if args.migrants_per_island is None or args.heldout_policy is None:
            p.error("non-smoke requires explicit --migrants-per-island and --heldout-policy; campaign policies are not frozen")
    if min(args.population, args.islands, args.generations, args.horizon, args.migration_interval,
           args.migrants_per_island, args.heldout_episodes) < 1:
        p.error("positive workload values required")
    if args.population % args.islands or args.population // args.islands < 2:
        p.error("population must divide islands and leave at least two individuals per island")
    if args.migrants_per_island > args.population // args.islands:
        p.error("migrants-per-island exceeds per-island population")
    return args


def adjacency(name, islands):
    if name == "single":
        return [[]]
    if name == "ring":
        return [[(i + 1) % islands] for i in range(islands)]
    if name == "bidirectional":
        return [sorted({(i - 1) % islands, (i + 1) % islands} - {i}) for i in range(islands)]
    return [[j for j in range(islands) if j != i] for i in range(islands)]


def migration_routes(links, budget, event_index):
    """Each ranked migrant is sent once; rotate destinations across events."""
    routes = []
    for source, targets in enumerate(links):
        if targets:
            for rank in range(budget):
                target = targets[(event_index * budget + rank) % len(targets)]
                routes.append({"source": source, "target": target, "migrant_rank": rank})
    return routes


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "schema_version": 2, "phase": 3, "status": "running", "smoke": args.smoke,
        "task": "MuJoCoMJXEnv Hopper", "algorithm": "MOEngine NSGA-II with asexual NeatEmitter mutation",
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "runs": [], "failures": [], "environment": {"python": sys.version, "executable": sys.executable},
        "limitations": [
            "Script-local orchestration, not existing scalar IslandEngineAdapter; production code unchanged.",
            "No NEAT speciation or sexual crossover; NEAT representation and structural/weight mutation are composed with NSGA-II.",
            "Fixed total outgoing budget per island/event; balanced neighbor rotation; capability demonstration, not topology superiority.",
            "Complexity is stored active connections (finite first field), not reachability-pruned expressed complexity.",
            "Instrumented rollout diagnostics record invalid/reset/termination counts; finite objectives alone do not imply valid dynamics.",
            "Simulation step slots count fixed scans including post-termination masked steps, not live environment transitions.",
            "Current script is a quantified capability runner, not a preregistered confirmation or architecture timing harness.",
        ],
    }
    sources = [str(Path(__file__).relative_to(ROOT)), "src/malthusjax/engine/mo/mo_engine.py",
               "src/malthusjax/core/genome/mo/population.py", "src/malthusjax/core/genome/mo/sorting.py",
               "packages/malthus_neat/src/malthus_neat/operators/neat_emitter.py",
               "packages/malthus_neat/src/malthus_neat/genomes/neat_genome.py",
               "packages/malthus_neat/src/malthus_neat/interpreters/neat_interpreter.py",
               "scripts/publication_rl.py", "src/malthusjax/core/fitness/composable/evaluators.py",
               "src/malthusjax/core/fitness/composable/environments.py"]
    report["source_sha256"] = {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}
    report["revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in ("jax", "jaxlib", "tensorneat", "mujoco", "malthus-neat"):
        try:
            report["environment"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["environment"][name] = None

    def save_report():
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from malthus_neat.genomes.neat_genome import NeatGenomeConfig
        from malthus_neat.interpreters.neat_interpreter import NeatInterpreter
        from malthus_neat.operators.neat_emitter import NeatEmitter

        from malthusjax.core.fitness.composable.base import ScalarOutput
        from malthusjax.core.fitness.composable.environments import MuJoCoMJXEnv
        from publication_rl import InstrumentedRLEvaluator as RLEvaluator
        from malthusjax.core.genome.mo.population import MOPopulation
        from malthusjax.engine.mo.mo_engine import MOEngine, MOEngineParams

        report["environment"]["backend"] = jax.default_backend()
        if args.smoke and jax.default_backend() != "cpu":
            raise RuntimeError("smoke requires CPU")
        env = MuJoCoMJXEnv.create(env_name="hopper", max_steps=args.horizon)
        if env.backend_pipeline is None or env.action_dim != 3:
            raise RuntimeError("Hopper creation fell back or has unexpected action dimensions; refusing substitution")
        config = NeatGenomeConfig(num_inputs=env.obs_dim, num_outputs=env.action_dim,
                                  max_nodes=args.max_nodes, max_conns=args.max_conns)
        if config.max_nodes < env.obs_dim + env.action_dim or config.max_conns < env.obs_dim * env.action_dim:
            raise ValueError("NEAT capacity insufficient for direct input-output initialization")
        interpreter = NeatInterpreter(genome_config=config)
        rl = RLEvaluator(env=env, interpreter=interpreter, output=ScalarOutput(maximize=True),
                         max_steps=args.horizon, num_eval_envs=1)
        report["environment"].update(hopper_obs_dim=env.obs_dim, hopper_action_dim=env.action_dim,
                                     hopper_backend_verified=True)

        class ObjectiveEvaluator:
            """Bridge existing scalar RL evaluation to the existing MO container."""
            requires_rng = True

            def __init__(self, environment_key):
                self.environment_key = environment_key

            def evaluate_population(self, population, rng=None):
                # Common random numbers: every training candidate/island/generation
                # uses the same reset/episode key within a seed block. Engine RNG
                # remains independent for variation. Heldout creates a new evaluator.
                fitness, infos = jax.vmap(lambda genome: rl.evaluate(
                    genome, genome, self.environment_key))(population.genes)
                evaluated = population.replace(fitness=fitness, info=infos)
                active = jnp.sum(jnp.isfinite(population.genes.conns[..., 0]), axis=1)
                objectives = jnp.stack([evaluated.fitness, active.astype(jnp.float32)], axis=1)
                return MOPopulation.from_evaluated(evaluated.replace(fitness=objectives), maximize=False)

        class ParentView:
            def __init__(self, population):
                self.population = population

            def select(self, key, batch_size):
                return self.population.select(key, batch_size).genes

        class EmitterBridge:
            def __init__(self, batch_size, island_index, num_islands):
                self.inner = NeatEmitter(genome_config=config, _batch_size=batch_size)
                self.island_index, self.num_islands = island_index, num_islands

            def init(self, *a, **kw):
                return self.inner.init(*a, **kw)

            def num_keys(self):
                return self.inner.num_keys()

            def ask(self, state, repertoire, keys, generation=0, params=None):
                # Unique mutation innovation slots across islands/generations.
                # Asexual mutation avoids undefined homology/crossover semantics.
                slot = generation * self.num_islands + self.island_index
                return self.inner.ask(state, ParentView(repertoire), keys, generation=slot, params=params)

            def tell(self, *a, **kw):
                return self.inner.tell(*a, **kw)

        def snapshot(pop):
            fit = np.asarray(pop.fitness)
            if fit.shape[1:] != (2,) or not np.isfinite(fit).all():
                raise ValueError("invalid multiobjective fitness")
            active_nodes = np.sum(np.isfinite(np.asarray(pop.genes.nodes)[..., 0]), axis=-1)
            active_connections = np.sum(np.isfinite(np.asarray(pop.genes.conns)[..., 0]), axis=-1)
            return {"objectives_minimize": fit.tolist(), "returns": (-fit[:, 0]).tolist(),
                    "active_nodes": active_nodes.tolist(), "active_connections": active_connections.tolist(),
                    "pareto_rank": np.asarray(pop.pareto_rank).tolist(),
                    "nondominated_count": int(np.sum(np.asarray(pop.pareto_rank) == 0)),
                    "rollout_diagnostics": {k: np.asarray(v).tolist() for k, v in (pop.info or {}).items() if k.startswith("rollout_")}}

        def slice_pop(pop, start, stop):
            updates = dict(genes=jax.tree_util.tree_map(lambda x: x[start:stop], pop.genes),
                           fitness=pop.fitness[start:stop],
                           info=jax.tree_util.tree_map(lambda x: x[start:stop], pop.info) if pop.info is not None else None)
            for field in ("pareto_rank", "crowding_distance"):
                if hasattr(pop, field):
                    updates[field] = getattr(pop, field)[start:stop]
            return pop.replace(**updates)

        for seed in args.seeds:
            root_key = jax.random.PRNGKey(seed)
            init_key, evolution_key, eval_key, training_env_key = jax.random.split(root_key, 4)
            objective = ObjectiveEvaluator(training_env_key)
            initial = config.init_population(init_key, args.population)
            leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(initial.genes)]
            init_hash = hashlib.sha256(b"".join(x.tobytes() for x in leaves)).hexdigest()
            np.savez(args.output / f"initial_seed_{seed}.npz", nodes=np.asarray(initial.genes.nodes), conns=np.asarray(initial.genes.conns))
            for topology in args.topologies:
                run_started = time.perf_counter()
                islands = 1 if topology == "single" else args.islands
                size = args.population // islands
                links = adjacency(topology, islands)
                # Retain full shared-array hash and deterministic contiguous slices.
                record = {"seed": seed, "topology": topology, "initial_population_hash": init_hash,
                          "partition": [[i * size, (i + 1) * size] for i in range(islands)],
                          "objective_convention": "minimize [-episodic_return, active_connection_count]",
                          "training_environment_key": np.asarray(training_env_key).tolist(),
                          "environment_policy": "fixed common episode/reset key across all candidates, generations and conditions within seed; heldout fresh shared episode keys",
                          "adjacency": links, "migration_events": [], "history": []}
                report["runs"].append(record)
                save_report()
                engines, states, steps = [], [], []
                for i in range(islands):
                    engine = MOEngine(emitter=EmitterBridge(size, i, islands), evaluator=objective,
                                      engine_params=MOEngineParams(pop_size=size, num_generations=args.generations))
                    # Equal total initialization; same environmental root key per block.
                    state = engine.init_state(jax.random.fold_in(evolution_key, i), slice_pop(initial, i * size, (i + 1) * size))
                    jax.block_until_ready(state)
                    engines.append(engine)
                    states.append(state)
                    steps.append(jax.jit(engine.step))
                for generation in range(args.generations):
                    states = [step(state)[0] for step, state in zip(steps, states)]
                    jax.block_until_ready(states)
                    completed = generation + 1
                    if islands > 1 and completed % args.migration_interval == 0 and completed < args.generations:
                        outgoing = [state.population.truncate(args.migrants_per_island) for state in states]
                        incoming = [[] for _ in states]
                        routes = migration_routes(links, args.migrants_per_island, len(record["migration_events"]))
                        for route in routes:
                            migrant = outgoing[route["source"]]
                            rank = route["migrant_rank"]
                            # merge consumes genes/fitness; recompute ranks on receiver pool.
                            incoming[route["target"]].append(slice_pop(migrant, rank, rank + 1))
                        next_states = []
                        for target, state in enumerate(states):
                            merged = state.population
                            for migrants in incoming[target]:
                                merged = merged.merge(migrants)
                            survived = merged.truncate(size)
                            next_states.append(state.replace(population=survived,
                                best_fitness=survived.fitness[0, 0],
                                best_genome=jax.tree_util.tree_map(lambda x: x[0], survived.genes)))
                        states = next_states
                        record["migration_events"].append({"generation": completed,
                            "transferred_individuals": len(routes),
                            "outgoing_budget_per_island": args.migrants_per_island, "routes": routes,
                            "duplicates": "retained; no deduplication",
                            "selection": "rank ascending, crowding descending; existing lexsort tie behavior", "survival": "receiver merge then NSGA-II truncate"})
                    record["history"].append({"generation": completed, "islands": [snapshot(st.population) for st in states]})
                    save_report()
                # Global display front is postprocessed, not a shared evolving archive.
                final_pop = states[0].population
                for state in states[1:]:
                    final_pop = final_pop.merge(state.population)
                record["final_training"] = snapshot(final_pop)
                np.savez(args.output / f"final_{topology}_seed_{seed}.npz", nodes=np.asarray(final_pop.genes.nodes), conns=np.asarray(final_pop.genes.conns), objectives=np.asarray(final_pop.fitness), pareto_rank=np.asarray(final_pop.pareto_rank))
                # Evaluate every final individual; explicitly budgeted whole population,
                # no heldout-based policy selection. Fresh episodes common per condition.
                heldout = []
                for episode in range(args.heldout_episodes):
                    evaluated = ObjectiveEvaluator(jax.random.fold_in(eval_key, episode)).evaluate_population(final_pop)
                    jax.block_until_ready(evaluated)
                    heldout.append(np.asarray(evaluated.fitness)[:, 0])
                mean_objectives = np.asarray(final_pop.fitness).copy()
                mean_objectives[:, 0] = np.mean(heldout, axis=0)
                heldout_pop = MOPopulation.from_evaluated(final_pop.replace(fitness=jnp.asarray(mean_objectives)))
                record["heldout"] = snapshot(heldout_pop)
                record["heldout_episode_returns"] = (-np.asarray(heldout)).tolist()
                record["generations_completed"] = args.generations
                record["training_evaluations_including_initialization"] = args.population * (args.generations + 1)
                record["heldout_evaluations"] = args.population * args.heldout_episodes
                record["simulation_step_slots"] = args.horizon * (record["training_evaluations_including_initialization"] + record["heldout_evaluations"])
                record["elapsed_seconds_including_preparation"] = time.perf_counter() - run_started
                record["status"] = "success"
                save_report()
                print(f"Phase3 {topology} seed={seed}: completed {args.generations} generations, {len(record['migration_events'])} migration events", flush=True)
        report["status"] = "success"
    except Exception as exc:
        report["status"] = "failed"
        report["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        import traceback
        (args.output / "traceback.txt").write_text(traceback.format_exc())
    save_report()
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"), "failures": report["failures"]}, indent=2))
    return 0 if report["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
