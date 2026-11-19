"""Run one registered problem with a native GA, EvoSAX, or QDax."""

import argparse
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from malthusjax.composer import Composer
from malthusjax.core.genome.real_genome import RealGenome


def main():
    # Composer loads local plugins by module stem. Reuse that registered module.
    from corridor import CorridorDemand, SensorLayout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("malthusjax", "evosax", "qdax"))
    backend = parser.parse_args().backend
    fitness = "research_corridor:worst_weight=0.30"
    options = {}
    if backend == "malthusjax":
        options["mutation"] = "gaussian:mutation_rate=0.05,mutation_strength=0.1"
    elif backend == "qdax":
        fitness += ",descriptor=layout"
        options.update(qdax_num_centroids=9, export_archive=True)

    result = Composer.create_default().quick_run(
        backend=backend,
        fitness=fitness,
        bounds=(0.0, 1.0),
        genome_length=2,
        pop_size=32,
        generations=35,
        seeds=(11, 22, 33, 44, 55),
        output_dir=Path("python-results") / backend,
        **options,
    )
    decoder = SensorLayout()
    objective = CorridorDemand()
    for run in result.runs:
        if run.status != "success":
            raise RuntimeError(run.error)
        print(run.seed, run.metrics["best_fitness"])
        if run.solution is not None:
            genome = RealGenome(values=jnp.asarray(run.solution["value"]))
            layout = decoder.apply(genome)
            assert math.isclose(
                float(objective.evaluate(layout)), run.solution["fitness"], abs_tol=1e-6
            )
            print("  sensor positions", layout.tolist())
        if "archive" in run.artifacts:
            with np.load(run.artifacts["archive"]) as archive:
                designs = [
                    decoder.apply(RealGenome(values=jnp.asarray(genes))).tolist()
                    for genes in archive["genotypes"]
                ]
                print("  occupied cells and designs", list(zip(archive["cell_index"], designs)))


if __name__ == "__main__":
    main()
