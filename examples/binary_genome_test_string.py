import time

from malthusjax.composer import Composer


def main() -> None:
    print("Testing Composer string configuration for Binary Genome...\n")
    composer = Composer()

    start_t = time.time()
    result = composer.quick_run(
        experiment_name="binary_sum_test",
        # New string spec matching what we implemented for GenomeCatalog
        genome="binary:length=50",
        fitness="binary_sum",
        selection="tournament:tournament_size=3",
        mutation="bitflip:mutation_rate=0.05",
        crossover="single_point",
        population_size=100,
        generations=100,
        seeds=(42,),
    )
    end_t = time.time()

    run_metrics = result.runs[0].metrics
    best_fit = run_metrics.get("best_fitness", "N/A")
    evals = run_metrics.get("evaluations", "N/A")

    print("\n--- TEST DONE ---")
    print(f"Experiment Name   : {result.name}")
    if best_fit != "N/A" and hasattr(best_fit, "item"):
        print(f"Global Best Fitness: {best_fit.item():.4f}")
    else:
        print(f"Global Best Fitness: {best_fit}")
    print(f"Total Evaluations : {evals}")
    print(f"Generations Run   : {len(result.runs[0].history)}")
    print(f"Execution Time    : {end_t - start_t:.4f}s")

    if result.runs[0].error:
        print(f"Run Error         : {result.runs[0].error}")

if __name__ == "__main__":
    main()
