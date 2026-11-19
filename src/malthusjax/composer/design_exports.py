"""Post-run selection and array packaging for optional design outputs."""

from typing import Any

import jax
import numpy as np


def selected_genotypes(genes: Any, selected: np.ndarray) -> dict[str, np.ndarray]:
    """Preserve each genome leaf's dtype in an NPZ-compatible mapping."""
    leaves = jax.tree.leaves(jax.device_get(genes))
    if len(leaves) == 1:
        return {"genotypes": np.asarray(leaves[0])[selected]}
    return {f"genotypes_{i}": np.asarray(leaf)[selected] for i, leaf in enumerate(leaves)}


def encoded_solution(genome: Any, fitness: float, scope: str) -> dict[str, Any]:
    from jax.flatten_util import ravel_pytree

    flat, _ = ravel_pytree(genome)
    return {
        "representation": "encoded_genome",
        "selection_scope": scope,
        "fitness": fitness,
        "value": flat.tolist(),
    }
