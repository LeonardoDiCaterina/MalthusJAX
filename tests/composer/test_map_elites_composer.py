import jax.numpy as jnp
from tensorneat.genome import DefaultConn, DefaultGenome, DefaultNode
from tensorneat.genome.operations.crossover import DefaultCrossover
from tensorneat.genome.operations.mutation import DefaultMutation

from malthusjax.composer.composer import Composer
from malthusjax.composer.strategies.core import MapElitesStrategy
from malthusjax.operators.emitters.tensorneat_emitter import TensorNeatEmitter


def test_composer_native_map_elites_tensorneat():
    composer = Composer.create_default()

    tn_genome = DefaultGenome(
        num_inputs=2,
        num_outputs=1,
        max_nodes=10,
        max_conns=20,
        node_gene=DefaultNode(),
        conn_gene=DefaultConn()
    )
    mutation = DefaultMutation()
    crossover = DefaultCrossover()

    emitter = TensorNeatEmitter(
        _batch_size=16,
        genome=tn_genome,
        mutation=mutation,
        crossover=crossover
    )

    strategy = MapElitesStrategy(
        emitter=emitter,
        num_descriptors=2,
        num_centroids=10
    )

    def objective_fn(nodes, conns):
        print("NODES SHAPE:", nodes.shape)
        fitnesses = jnp.sum(~jnp.isnan(nodes[:, :, 0]), axis=1).astype(jnp.float32)
        desc_x = jnp.clip(fitnesses / 10.0, 0.0, 1.0)
        desc_y = jnp.clip((fitnesses ** 2) / 100.0, 0.0, 1.0)
        descriptors = jnp.stack([desc_x, desc_y], axis=-1)
        return fitnesses, descriptors

    result = composer.quick_run(
        strategy=strategy,
        objective_function=objective_fn,
        pop_size=16,
        generations=2,
        seeds=[1]
    )

    # Check that a result was produced
    assert result is not None
    assert len(result.runs) == 1

    run = result.runs[0]
    # In QD, the state holds the repertoire which tracks max fitness
    # For now just checking it completed generations
    assert len(run.history) > 0


def test_native_vs_qdax_benchmarking():
    from malthusjax.composer.strategies.core import QDAXStrategy
    from malthusjax.operators.emitters.genetic import GeneticMutationEmitter
    from malthusjax.operators.mutation import GaussianMutation
    composer = Composer.create_default()

    pipelines = {
        "Native_MAP_Elites": {
            "strategy": MapElitesStrategy(
                emitter=GeneticMutationEmitter(
                    mutation=GaussianMutation(mutation_rate=0.5, mutation_strength=0.1),
                    genome_config=None,
                    _batch_size=20
                ),
                num_descriptors=2,
                num_centroids=10
            ),
            "fitness": "bbob:rosenbrock:dim=10",
            "pop_size": 20,
        },
        "QDAX_MAP_Elites": {
            "strategy": QDAXStrategy(
                strategy_cls="MAPElites",
                num_descriptors=2,
                num_centroids=10,
                mutation_sigma=0.1
            ),
            "fitness": "bbob:rosenbrock:dim=10",
            "pop_size": 20,
        }
    }

    result = composer.compare(
        pipelines=pipelines,
        generations=2,
        seeds=[1],
        maximize=True,
        bounds=(-5.0, 5.0)
    )

    assert result is not None
    assert "Native_MAP_Elites" in result.pipelines
    assert "QDAX_MAP_Elites" in result.pipelines
    assert len(result.pipelines["Native_MAP_Elites"].runs) == 1
    assert len(result.pipelines["QDAX_MAP_Elites"].runs) == 1

