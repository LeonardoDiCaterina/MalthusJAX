from malthusjax.composer.catalog import OperatorCatalog
from malthusjax.composer.composer import Composer
from malthusjax.composer.strategies.core import MapElitesStrategy
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.operators.emitters.genetic import GeneticMutationEmitter


def main():
    composer = Composer.create_default()

    # 1. Create equivalent MalthusJAX native emitter
    cat = OperatorCatalog()
    mutation = cat.get("gaussian:mutation_rate=1.0,mutation_strength=0.1")

    # QDAX MAP-Elites default just mutates. We match it with a GeneticMutationEmitter
    native_emitter = GeneticMutationEmitter(
        _batch_size=100,  # Matches pop_size for QDAX parity
        mutation=mutation,
        genome_config=RealGenomeConfig(shape=(10,), bounds=(-5.0, 5.0)),
    )

    print("Running Benchmark: Native MalthusJAX MAP-Elites vs QDAX MAP-Elites")

    comparison = composer.compare(
        pipelines={
            "MalthusJAX Native MAP-Elites": dict(
                strategy=MapElitesStrategy(emitter=native_emitter, num_centroids=50)
            ),
            "QDAX MAP-Elites": dict(
                backend="qdax",
                qdax_strategy="MAPElites",
                qdax_num_centroids=50,
                qdax_mutation_sigma=0.1,
            ),
        },
        fitness="rosenbrock:dim=10",
        pop_size=100,
        generations=500,
        seeds=(42, 43, 44),
        shared_initial_population=False,
        maximize=False,  # Minimize rosenbrock
        bounds=(-5.0, 5.0),
    )

    print("\nBenchmark Results Summary:")
    import pprint

    pprint.pprint(comparison.summary_table())


if __name__ == "__main__":
    main()
