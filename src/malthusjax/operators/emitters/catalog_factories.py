"""Factories and lambda registries for QDAX emitters integration with OperatorCatalog."""

from typing import Any, Callable, Dict, Tuple

import jax

from malthusjax.core.genome.real_genome import RealGenomeConfig

# We import the QDAXReplicaMixingEmitter for the MalthusJAX engine
from malthusjax.operators.emitters.qdax_replica import QDAXReplicaMixingEmitter

# ---------------------------------------------------------------------------
# PURE LAMBDA REGISTRIES FOR QDAX EMITTERS
# These lambdas must natively support batched inputs (jnp.ndarray) and take
# a single PRNGKey for the entire batch.
# ---------------------------------------------------------------------------

QDAX_MUTATION_LAMBDAS: Dict[str, Callable] = {
    "gaussian": lambda x, key, sigma=0.1: x + jax.random.normal(key, x.shape) * sigma,
    # Additional pure lambda mutations (like polynomial) can be added here
}

QDAX_VARIATION_LAMBDAS: Dict[str, Callable] = {
    "none": lambda x1, x2, key: x1,
    # Additional pure lambda variations (like iso_dd) can be added here
}


def _parse_lambda_spec(spec: str) -> Tuple[str, Dict[str, Any]]:
    """Helper to parse a spec like 'gaussian:sigma=0.2'."""
    spec = spec.strip()
    if ":" not in spec:
        return spec, {}
    name, params_str = spec.split(":", 1)
    params = {}
    for param_pair in params_str.split(","):
        k, v = param_pair.split("=")
        try:
            params[k.strip()] = float(v.strip())
        except ValueError:
            params[k.strip()] = v.strip()
    return name.strip(), params


# ---------------------------------------------------------------------------
# EMITTER FACTORIES
# ---------------------------------------------------------------------------


def build_qdax_replica_emitter(**kwargs: Any) -> QDAXReplicaMixingEmitter:
    """Builds the MalthusJAX replica of the QDAX MixingEmitter."""
    mut_spec = kwargs.get("mutation", "gaussian")
    var_spec = kwargs.get("crossover", "none")

    mut_name, mut_kwargs = _parse_lambda_spec(mut_spec)
    var_name, var_kwargs = _parse_lambda_spec(var_spec)

    if mut_name not in QDAX_MUTATION_LAMBDAS:
        raise ValueError(f"Unknown QDAX mutation lambda: {mut_name}")
    if var_name not in QDAX_VARIATION_LAMBDAS:
        raise ValueError(f"Unknown QDAX variation lambda: {var_name}")

    mut_base_fn = QDAX_MUTATION_LAMBDAS[mut_name]
    var_base_fn = QDAX_VARIATION_LAMBDAS[var_name]

    # Bind kwargs
    def mut_fn(x, key):
        return mut_base_fn(x, key, **mut_kwargs)
    def var_fn(x1, x2, key):
        return var_base_fn(x1, x2, key, **var_kwargs)

    bounds = kwargs.get("bounds", (-5.0, 5.0))
    genome_length = kwargs.get("genome_length", 10)
    batch_size = kwargs.get("batch_size", 50)

    return QDAXReplicaMixingEmitter(
        mutation_fn=mut_fn,
        variation_fn=var_fn,
        variation_percentage=float(kwargs.get("variation_percentage", 0.5)),
        _batch_size=int(batch_size),
        genome_config=RealGenomeConfig(bounds=bounds, shape=(int(genome_length),)),
    )


def build_qdax_native_emitter(**kwargs: Any) -> Any:
    """Builds the actual official qdax.core.emitters.standard_emitters.MixingEmitter."""
    # We import dynamically so qdax is not a hard dependency unless requested
    from qdax.core.emitters.standard_emitters import MixingEmitter

    mut_spec = kwargs.get("mutation", "gaussian")
    var_spec = kwargs.get("crossover", "none")

    mut_name, mut_kwargs = _parse_lambda_spec(mut_spec)
    var_name, var_kwargs = _parse_lambda_spec(var_spec)

    if mut_name not in QDAX_MUTATION_LAMBDAS:
        raise ValueError(f"Unknown QDAX mutation lambda: {mut_name}")
    if var_name not in QDAX_VARIATION_LAMBDAS:
        raise ValueError(f"Unknown QDAX variation lambda: {var_name}")

    mut_base_fn = QDAX_MUTATION_LAMBDAS[mut_name]
    var_base_fn = QDAX_VARIATION_LAMBDAS[var_name]

    # Bind kwargs
    def mut_fn(x, key):
        return mut_base_fn(x, key, **mut_kwargs)
    def var_fn(x1, x2, key):
        return var_base_fn(x1, x2, key, **var_kwargs)

    batch_size = kwargs.get("batch_size", 50)

    return MixingEmitter(
        mutation_fn=mut_fn,
        variation_fn=var_fn,
        variation_percentage=float(kwargs.get("variation_percentage", 0.5)),
        batch_size=int(batch_size),
    )


def build_genetic_mixing_emitter(**kwargs: Any) -> Any:
    """Builds a GeneticMixingEmitter using native MalthusJAX operators."""
    from malthusjax.composer.catalog import OperatorCatalog
    from malthusjax.operators.emitters.genetic import GeneticMixingEmitter

    catalog = OperatorCatalog()

    mut_spec = kwargs.get("mutation", "gaussian")
    var_spec = kwargs.get("crossover", "none")

    sub_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in [
            "mutation",
            "crossover",
            "variation_percentage",
            "batch_size",
            "genome_length",
            "bounds",
        ]
    }

    mut_op = catalog.get(mut_spec, **sub_kwargs)
    var_op = catalog.get(var_spec, **sub_kwargs)

    bounds = kwargs.get("bounds", (-5.0, 5.0))
    genome_length = kwargs.get("genome_length", 10)
    batch_size = kwargs.get("batch_size", 50)

    return GeneticMixingEmitter(
        mutation=mut_op,
        crossover=var_op,
        variation_percentage=float(kwargs.get("variation_percentage", 0.5)),
        _batch_size=int(batch_size),
        genome_config=RealGenomeConfig(bounds=bounds, shape=(int(genome_length),)),
    )
