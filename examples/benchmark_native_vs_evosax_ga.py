from malthusjax.composer.composer import Composer


def main():
    composer = Composer.create_default()

    print("Running GA Benchmark: Native MalthusJAX vs Evosax SimpleGA")

    # We will use a standard continuous fitness function like rosenbrock
    comparison = composer.compare(
        pipelines={
            "MalthusJAX Native GA": dict(
                crossover="simulated_binary:eta=2.0",
                mutation="polynomial:mutation_rate=0.1,eta=20.0",
            ),
            "Evosax SimpleGA": dict(
                backend="evosax",
                evosax_strategy="SimpleGA",
            ),
        },
        fitness="rosenbrock:dim=10",
        pop_size=200,
        generations=100,
        seeds=(42, 43, 44),
        shared_initial_population=True,
        maximize=False,  # Minimize rosenbrock
        bounds=(-5.0, 5.0),
    )

    print("\nBenchmark Results Summary:")
    import pprint

    pprint.pprint(comparison.summary_table())


if __name__ == "__main__":
    main()
