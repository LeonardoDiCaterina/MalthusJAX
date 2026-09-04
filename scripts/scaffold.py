#!/usr/bin/env python3
"""
MalthusJAX Scaffolding Tool

Generates boilerplate code for building JAX-compliant components for MalthusJAX.
"""

import argparse
import os
import re
import string

from pathlib import Path


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


# --- IMPLEMENTATION TEMPLATES ---

SELECTION_TEMPLATE = string.Template('''import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseSelection
from malthusjax.composer import register_selection

@register_selection("${key}", override=True)
@struct.dataclass
class ${name}(BaseSelection):
    """A custom selection operator."""
    
    num_selections: int = struct.field(pytree_node=False)

    def __call__(self, rng: jax.Array, population: jax.Array, fitness: jax.Array, **kwargs) -> jax.Array:
        # TODO: Implement your selection logic here.
        # Return the indices of the selected individuals.
        return jnp.arange(self.num_selections)
''')

MUTATION_TEMPLATE = string.Template('''import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.operators.base import BaseMutation
from malthusjax.composer import register_mutation

@register_mutation("${key}", override=True, compatible_genomes=["real", "continuous"])
@struct.dataclass
class ${name}(BaseMutation):
    """A custom mutation operator."""
    
    mutation_rate: float = 0.1

    def __call__(self, rng: jax.Array, genome: jax.Array, **kwargs) -> jax.Array:
        # TODO: Implement your mutation logic here
        return genome
''')

CROSSOVER_TEMPLATE = string.Template('''import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

from malthusjax.operators.base import BaseCrossover
from malthusjax.composer import register_crossover

@register_crossover("${key}", override=True, compatible_genomes=["real", "continuous"])
@struct.dataclass
class ${name}(BaseCrossover):
    """A custom crossover operator."""
    
    crossover_rate: float = 0.9

    def __call__(self, rng: jax.Array, parent1: jax.Array, parent2: jax.Array, **kwargs) -> Tuple[jax.Array, jax.Array]:
        # TODO: Implement your crossover logic here
        return parent1, parent2
''')

ENGINE_TEMPLATE = string.Template('''from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.engine.base import AbstractEngine, EngineState
from malthusjax.composer import register_engine

@struct.dataclass
class ${name}State(EngineState):
    """Custom state for ${name}."""
    # TODO: Add custom state fields
    pass

@register_engine("${key}", override=True)
@struct.dataclass
class ${name}(AbstractEngine):
    """A custom evolutionary engine."""
    
    def init(self, rng: jax.Array) -> ${name}State:
        # TODO: Initialize engine state
        return ${name}State(
            generation=0,
            best_fitness=-jnp.inf,
            best_genome=jnp.zeros(()),
        )

    def step(self, rng: jax.Array, state: ${name}State, population: jax.Array, fitness: jax.Array) -> Tuple[${name}State, jax.Array]:
        # TODO: Implement a single generation step
        return state, population
''')

FITNESS_TEMPLATE = string.Template('''import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.base import BaseEvaluator
from malthusjax.composer import register_fitness

@register_fitness("${key}", override=True)
@struct.dataclass
class ${name}(BaseEvaluator):
    """A custom fitness evaluator."""

    def evaluate(self, rng: jax.Array, genome: jax.Array) -> jax.Array:
        # TODO: Implement fitness evaluation
        # Note: This operates on a single genome. MalthusJAX handles `vmap` internally.
        return jnp.sum(genome)
''')

GENOME_TEMPLATE = string.Template('''from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome
from malthusjax.composer import register_genome

@register_genome("${key}", override=True)
@struct.dataclass
class ${name}(BaseGenome):
    """A custom genome configuration."""
    
    shape: Tuple[int, ...] = struct.field(pytree_node=False)

    def initialize(self, rng: jax.Array) -> jax.Array:
        # TODO: Implement genome initialization
        return jax.random.normal(rng, self.shape)
''')

# --- TEST TEMPLATES ---

TEST_TEMPLATE_GENERIC = string.Template('''import jax
import jax.numpy as jnp
import pytest
from flax import struct

# Assuming the output is in the `plugins/` directory
from plugins.${module_name} import ${name}

def test_${module_name}_instantiation():
    """Verify that the component instantiates correctly and is a frozen dataclass."""
    # Note: adjust kwargs if your component has required positional arguments without defaults
    component = ${name}(${dummy_args})
    assert isinstance(component, ${name})

def test_${module_name}_call():
    """Verify that the component executes without JAX tracer errors."""
    component = ${name}(${dummy_args})
    rng = jax.random.PRNGKey(0)
    
    # TODO: provide correct dummy shapes to the call/step function
    ${dummy_call}
''')

DUMMY_ARGS = {
    "selection": "num_selections=5",
    "mutation": "",
    "crossover": "",
    "engine": "",
    "fitness": "",
    "genome": "shape=(10,)"
}

DUMMY_CALLS = {
    "selection": "population = jax.random.normal(rng, (10, 5))\n    fitness = jnp.zeros(10)\n    indices = component(rng, population, fitness)\n    assert indices.shape == (5,)",
    "mutation": "genome = jax.random.normal(rng, (5,))\n    mutated = component(rng, genome)\n    assert mutated.shape == (5,)",
    "crossover": "p1 = jax.random.normal(rng, (5,))\n    p2 = jax.random.normal(rng, (5,))\n    c1, c2 = component(rng, p1, p2)\n    assert c1.shape == p1.shape",
    "engine": "state = component.init(rng)\n    pop = jnp.zeros((10, 5))\n    fit = jnp.zeros(10)\n    new_state, new_pop = component.step(rng, state, pop, fit)",
    "fitness": "genome = jnp.zeros(5)\n    fit = component.evaluate(rng, genome)\n    assert fit.shape == ()",
    "genome": "genome = component.initialize(rng)\n    assert genome.shape == (10,)"
}


TEMPLATES = {
    "selection": SELECTION_TEMPLATE,
    "mutation": MUTATION_TEMPLATE,
    "crossover": CROSSOVER_TEMPLATE,
    "engine": ENGINE_TEMPLATE,
    "fitness": FITNESS_TEMPLATE,
    "genome": GENOME_TEMPLATE,
}


def main():
    parser = argparse.ArgumentParser(description="MalthusJAX Component Scaffolder")
    parser.add_argument("--type", "-t", required=True, choices=TEMPLATES.keys(), help="Type of component to scaffold")
    parser.add_argument("--name", "-n", required=True, help="Class name (e.g., QuantumMutation)")
    parser.add_argument("--key", "-k", required=True, help="Registry key (e.g., quantum)")
    parser.add_argument("--out", "-o", default="plugins", help="Output directory for implementation")
    parser.add_argument("--test-out", "-to", default="tests/plugins", help="Output directory for tests")
    
    args = parser.parse_args()
    
    # Resolve paths
    out_dir = Path(args.out)
    test_out_dir = Path(args.test_out)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)
    
    module_name = to_snake_case(args.name)
    impl_file = out_dir / f"{module_name}.py"
    test_file = test_out_dir / f"test_{module_name}.py"
    
    if impl_file.exists():
        print(f"Error: {impl_file} already exists!")
        return 1
        
    if test_file.exists():
        print(f"Error: {test_file} already exists!")
        return 1

    # Render Implementation
    impl_content = TEMPLATES[args.type].substitute(name=args.name, key=args.key)
    impl_file.write_text(impl_content)
    
    # Render Test Boilerplate
    test_content = TEST_TEMPLATE_GENERIC.substitute(
        name=args.name,
        module_name=module_name,
        dummy_args=DUMMY_ARGS[args.type],
        dummy_call=DUMMY_CALLS[args.type]
    )
    test_file.write_text(test_content)
    
    print(f"Successfully scaffolded {args.type} '{args.name}'")
    print(f"  Implementation: {impl_file}")
    print(f"  Test:           {test_file}")
    return 0


if __name__ == "__main__":
    exit(main())
