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

ADAPTER_TEMPLATE = string.Template('''from typing import Any, Tuple, Dict, Callable
import jax
import jax.numpy as jnp
import chex

from malthusjax.composer.adapters import adapter, EvalMode
from malthusjax.composer.adapters.metrics import MetricSpec

ADAPTER_METRICS = [
    MetricSpec(name="best_fitness", source="best_fitness", is_objective_value=True),
    MetricSpec(name="mean_fitness", source="mean_fitness", is_objective_value=True),
]

@adapter(
    framework="${key}",
    state_mapping={"init": "_adapter_init", "step": "_adapter_step"},
    eval_translators={EvalMode.NATIVE: lambda *args: None, EvalMode.MALTHUSJAX: lambda *args: None},
    metrics_catalog=ADAPTER_METRICS,
)
class ${name}:
    """Universal Adapter for ${key}."""

    def _adapter_init(self, strategy: Any, key: chex.PRNGKey, params: Any, pop_init: Any = None) -> Any:
        # TODO: Initialize your framework's state here
        return None

    def _adapter_step(
        self,
        strategy: Any,
        state: Any,
        key: chex.PRNGKey,
        params: Any,
        evaluator: Any,
        eval_translator: Callable[..., Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        # TODO: Implement a single step of your framework
        # 1. Ask
        # 2. Evaluate using eval_translator
        # 3. Tell
        # 4. Return new state and metrics dict
        return state, {}
''')


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

    def _select_one(self, rng: jax.Array, population: jax.Array, fitness: jax.Array) -> jax.Array:
        # TODO: Implement your selection logic here for ONE selection.
        # Return the indices of the selected individual.
        return jnp.zeros((), dtype=jnp.int32)
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

    def _mutate_one(self, rng: jax.Array, genome: jax.Array) -> jax.Array:
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

    def _recombine_one(self, rng: jax.Array, parent1: jax.Array, parent2: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # TODO: Implement your crossover logic here
        return parent1, parent2
''')

ENGINE_TEMPLATE = string.Template('''from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.engine.base import AbstractEngine, EngineState, AbstractEngineParams
from malthusjax.composer import register_engine

@struct.dataclass
class ${name}State(EngineState):
    """Custom state for ${name}."""
    pass

@register_engine("${key}", override=True)
@struct.dataclass
class ${name}(AbstractEngine):
    """A custom evolutionary engine."""
    
    engine_params: AbstractEngineParams
    
    @property
    def maximize(self) -> bool:
        return True

    def init_state(self, rng_key: jax.Array) -> ${name}State:
        # TODO: Initialize engine state
        return ${name}State(
            generation=0,
            best_fitness=-jnp.inf,
            best_genome=jnp.zeros(()),
            population=None, # Define initial pop
            rng_key=rng_key
        )

    def step(self, state: ${name}State) -> Tuple[${name}State, Any]:
        # TODO: Implement a single generation step
        output = None
        return state, output
''')

FITNESS_TEMPLATE = string.Template('''import jax
import jax.numpy as jnp
from flax import struct
from typing import Any

from malthusjax.core.fitness.base import BaseEvaluator, BaseEvaluatorConfig
from malthusjax.composer import register_fitness

@register_fitness("${key}", override=True)
@struct.dataclass
class ${name}(BaseEvaluator):
    """A custom fitness evaluator."""

    config: BaseEvaluatorConfig
    data: Any = struct.field(pytree_node=False, default=None)

    def evaluate(self, genome: jax.Array) -> jax.Array:
        # TODO: Implement fitness evaluation
        # Note: This operates on a single genome. MalthusJAX handles `vmap` internally.
        return jnp.sum(genome)
''')

GENOME_TEMPLATE = string.Template('''from typing import Any, Tuple, Type

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome
from malthusjax.composer import register_genome

@register_genome("${key}", override=True)
@struct.dataclass
class ${name}(BaseGenome):
    """A custom genome configuration."""

    values: chex.Array
    
    @classmethod
    def random_init(cls: Type["${name}"], key: chex.PRNGKey, config: Any) -> "${name}":
        # TODO: Initialize random genome values
        shape = (10,)
        return cls(values=jax.random.normal(key, shape))

    def distance(self, other: BaseGenome, metric: str) -> chex.Numeric:
        # TODO: Implement distance metric
        return jnp.sum(jnp.abs(self.values - other.values))

    def autocorrect(self, config: Any) -> "${name}":
        # TODO: Enforce constraints
        return self

    @property
    def size(self) -> int:
        return self.values.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @classmethod
    def from_tensor(cls: Type["${name}"], arr: chex.Array, config: Any = None) -> "${name}":
        return cls(values=arr)
''')

POPULATION_TEMPLATE = string.Template('''from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation

@struct.dataclass
class ${name}(BasePopulation):
    """A custom population container."""
    # Add your custom fields here
    # Example: custom_metadata: chex.Array = struct.field(default_factory=lambda: jnp.array([]))
    pass
''')

# --- TEST TEMPLATES ---

TEST_ADAPTER = string.Template('''import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import AdapterComplianceSuite

class Test${name}(AdapterComplianceSuite):
    @pytest.fixture
    def component(self):
        # The adapter decorator changes the class signature.
        # We need to instantiate it with dummy args expected by the universal engine base.
        return ${name}(
            strategy=None,
            params=None,
            pop_size=10,
            num_generations=5,
        )
''')


TEST_SELECTION = string.Template('''import jax
import jax.numpy as jnp
import pytest
from malthusjax.core.base import BasePopulation
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import SelectionComplianceSuite

class Test${name}(SelectionComplianceSuite):
    @pytest.fixture
    def component(self):
        return ${name}(${dummy_args})
    
    @pytest.fixture
    def mock_population(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return BasePopulation(genes=genes, fitness=fitness)
''')

TEST_MUTATION = string.Template('''import jax
import jax.numpy as jnp
import pytest
from malthusjax.core.base import BasePopulation
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import MutationComplianceSuite

class Test${name}(MutationComplianceSuite):
    @pytest.fixture
    def component(self):
        return ${name}(${dummy_args})
    
    @pytest.fixture
    def mock_population(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return BasePopulation(genes=genes, fitness=fitness)
''')

TEST_CROSSOVER = string.Template('''import jax
import jax.numpy as jnp
import pytest
from malthusjax.core.base import BasePopulation
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import CrossoverComplianceSuite

class Test${name}(CrossoverComplianceSuite):
    @pytest.fixture
    def component(self):
        return ${name}(${dummy_args})
    
    @pytest.fixture
    def mock_population(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return BasePopulation(genes=genes, fitness=fitness)
''')

TEST_ENGINE = string.Template('''import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import EngineComplianceSuite
from malthusjax.engine.base import AbstractEngineParams

class Test${name}(EngineComplianceSuite):
    @pytest.fixture
    def component(self):
        params = AbstractEngineParams(pop_size=10, num_generations=5)
        return ${name}(engine_params=params)
''')

TEST_FITNESS = string.Template('''import jax
import jax.numpy as jnp
import pytest
from malthusjax.core.base import BasePopulation
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import EvaluatorComplianceSuite
from malthusjax.core.fitness.base import BaseEvaluatorConfig

class Test${name}(EvaluatorComplianceSuite):
    @pytest.fixture
    def component(self):
        config = BaseEvaluatorConfig()
        return ${name}(config=config, data=None)

    @pytest.fixture
    def mock_population(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return BasePopulation(genes=genes, fitness=fitness)
''')

TEST_GENOME = string.Template('''import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import GenomeComplianceSuite

class Test${name}(GenomeComplianceSuite):
    @pytest.fixture
    def component(self):
        rng = jax.random.PRNGKey(0)
        return ${name}.random_init(rng, config=None)
''')

TEST_POPULATION = string.Template('''import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import PopulationComplianceSuite

class Test${name}(PopulationComplianceSuite):
    @pytest.fixture
    def component(self):
        genes = jnp.zeros((10, 5))
        fitness = jnp.zeros(10)
        return ${name}(genes=genes, fitness=fitness)
''')

DUMMY_ARGS = {
    "selection": "num_selections=5",
    "mutation": "",
    "crossover": "",
    "engine": "",
    "fitness": "",
    "genome": "values=jnp.zeros(10)",
    "population": "",
    "adapter": ""
}

TEST_TEMPLATES = {
    "selection": TEST_SELECTION,
    "mutation": TEST_MUTATION,
    "crossover": TEST_CROSSOVER,
    "engine": TEST_ENGINE,
    "fitness": TEST_FITNESS,
    "genome": TEST_GENOME,
    "population": TEST_POPULATION,
    "adapter": TEST_ADAPTER,
}

TEMPLATES = {
    "selection": SELECTION_TEMPLATE,
    "mutation": MUTATION_TEMPLATE,
    "crossover": CROSSOVER_TEMPLATE,
    "engine": ENGINE_TEMPLATE,
    "fitness": FITNESS_TEMPLATE,
    "genome": GENOME_TEMPLATE,
    "population": POPULATION_TEMPLATE,
    "adapter": ADAPTER_TEMPLATE,
}


def main():
    parser = argparse.ArgumentParser(description="MalthusJAX Component Scaffolder")
    parser.add_argument("--type", "-t", required=True, choices=TEMPLATES.keys(), help="Type of component to scaffold")
    parser.add_argument("--name", "-n", required=True, help="Class name (e.g., QuantumMutation)")
    parser.add_argument("--key", "-k", required=True, help="Registry key (e.g., quantum)")
    parser.add_argument("--impl-dir", "-o", default="plugins", help="Output directory for implementation")
    parser.add_argument("--test-dir", "-to", default="tests/plugins", help="Output directory for tests")
    
    args = parser.parse_args()
    
    # Resolve paths
    out_dir = Path(args.impl_dir)
    test_out_dir = Path(args.test_dir)
    
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
    
    import_path = str(out_dir).replace('/', '.')
    
    # Render Test Boilerplate
    test_content = TEST_TEMPLATES[args.type].substitute(
        name=args.name,
        module_name=module_name,
        import_path=import_path,
        dummy_args=DUMMY_ARGS[args.type]
    )
    test_file.write_text(test_content)
    
    print(f"Successfully scaffolded {args.type} '{args.name}'")
    print(f"  Implementation: {impl_file}")
    print(f"  Test:           {test_file}")
    return 0


if __name__ == "__main__":
    exit(main())
