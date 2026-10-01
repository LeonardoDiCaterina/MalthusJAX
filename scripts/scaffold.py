#!/usr/bin/env python3
"""
MalthusJAX Scaffolding Tool

Generates boilerplate code for building JAX-compliant components for MalthusJAX.
"""

import argparse
import re
import string
from pathlib import Path


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


# --- IMPLEMENTATION TEMPLATES ---

ADAPTER_TEMPLATE = string.Template('''from typing import Any, Tuple, Dict, Callable, Optional
import jax
import jax.numpy as jnp
import chex

from malthusjax.composer.adapters import adapter, EvalMode
from malthusjax.composer.adapters.metrics import MetricSpec
from malthusjax.core.logger import StepLoggingConfig, get_logger

logger = get_logger("composer.adapters.${key}")

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

    def _adapter_init(
        self,
        strategy: Any,
        key: chex.PRNGKey,
        params: Any,
        pop_init: Any = None,
        step_logging: Optional[StepLoggingConfig] = None,
    ) -> Any:
        logger.debug("Initializing ${name} adapter for strategy %s", strategy)
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
from malthusjax.core.logger import get_logger

logger = get_logger("operators.selection.${key}")

@register_selection("${key}", override=True)
@struct.dataclass
class ${name}(BaseSelection):
    """A custom selection operator."""

    num_selections: int = struct.field(pytree_node=False)

    def set_input_length(self, length: int) -> "${name}":
        logger.debug("Configuring ${name} selection count: %d", length)
        return self.replace(num_selections=length)

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
from malthusjax.core.logger import get_logger

logger = get_logger("operators.mutation.${key}")

@register_mutation("${key}", override=True, compatible_genomes=["real", "continuous"])
@struct.dataclass
class ${name}(BaseMutation):
    """A custom mutation operator."""

    mutation_rate: float = 0.1

    def set_input_length(self, length: int) -> "${name}":
        logger.debug("Configuring ${name} input length: %d", length)
        return self

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
from malthusjax.core.logger import get_logger

logger = get_logger("operators.crossover.${key}")

@register_crossover("${key}", override=True, compatible_genomes=["real", "continuous"])
@struct.dataclass
class ${name}(BaseCrossover):
    """A custom crossover operator."""

    crossover_rate: float = 0.9

    def set_input_length(self, length: int) -> "${name}":
        logger.debug("Configuring ${name} input length: %d", length)
        return self

    def _recombine_one(self, rng: jax.Array, parent1: jax.Array, parent2: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # TODO: Implement your crossover logic here
        return parent1, parent2
''')

EMITTER_TEMPLATE = string.Template('''from typing import Any, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.operators.emitters.base import BaseEmitter, EmitterState
from malthusjax.composer import register_emitter
from malthusjax.core.logger import get_logger

logger = get_logger("operators.emitters.${key}")

@struct.dataclass
class ${name}State(EmitterState):
    """Internal state for ${name}."""
    pass

@register_emitter("${key}", override=True)
@struct.dataclass
class ${name}(BaseEmitter):
    """A custom quality-diversity emitter."""

    _batch_size: int = struct.field(pytree_node=False, default=32)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_keys_per_atomic_operation(self) -> int:
        return 1

    def set_input_length(self, length: int) -> "${name}":
        logger.debug("Configuring ${name} batch size: %d", length)
        return self.replace(_batch_size=length)

    def init(
        self, key: chex.Array, initial_population: BasePopulation[Any], params: Any = None
    ) -> Optional[EmitterState]:
        logger.debug("Initializing ${name} emitter state")
        return ${name}State()

    def ask(
        self,
        state: Optional[EmitterState],
        repertoire: Any,
        keys: chex.Array,
        generation: int = 0,
        params: Any = None,
    ) -> Tuple[BasePopulation[Any], Optional[EmitterState]]:
        # TODO: Implement offspring generation logic from repertoire
        genes = jnp.zeros((self.batch_size, 10))
        fitness = jnp.zeros(self.batch_size)
        offspring = BasePopulation(genes=genes, fitness=fitness)
        return offspring, state
''')

ENGINE_TEMPLATE = string.Template('''from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.engine.base import AbstractEngine, EngineState, AbstractEngineParams
from malthusjax.composer import register_engine
from malthusjax.core.logger import StepLoggingConfig, get_logger, _host_log_step, _host_log_nan_anomaly

logger = get_logger("engine.${key}")

@struct.dataclass
class ${name}State(EngineState):
    """Custom state for ${name}."""
    pass

@register_engine("${key}", override=True)
@struct.dataclass
class ${name}(AbstractEngine):
    """A custom evolutionary engine with unified logging support."""

    engine_params: AbstractEngineParams

    @property
    def maximize(self) -> bool:
        return True

    def init_state(self, rng_key: jax.Array) -> ${name}State:
        logger.debug("Initializing ${name} state")
        # TODO: Initialize engine state
        return ${name}State(
            generation=0,
            best_fitness=-jnp.inf,
            best_genome=jnp.zeros(()),
            population=None, # Define initial pop
            rng_key=rng_key
        )

    def step(self, state: ${name}State) -> Tuple[${name}State, Any]:
        # TODO: Implement a single generation step.
        # To bridge step telemetry to Python host logging:
        # if self.engine_params.step_logging and self.engine_params.step_logging.is_active():
        #     jax.lax.cond(
        #         (state.generation % self.engine_params.step_logging.log_interval) == 0,
        #         lambda: jax.debug.callback(
        #             _host_log_step, state.generation, state.best_fitness, state.best_fitness, self.engine_params.step_logging.logger_name
        #         ),
        #         lambda: None,
        #     )
        output = None
        return state, output
''')

EVALUATOR_TEMPLATE = string.Template('''import jax
import jax.numpy as jnp

# Choose the appropriate core evaluator
from malthusjax.core.fitness.composable.evaluators import (
    OptimizationEvaluator, SupervisedEvaluator, RLEvaluator
)
from malthusjax.core.fitness.composable.base import ScalarOutput, IdentityTransform
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.logger import get_logger

logger = get_logger("core.fitness.evaluator")

def create_${module_name}() -> OptimizationEvaluator:
    """Builds and returns a configured composable evaluator."""
    logger.debug("Building composable evaluator: %s", "${module_name}")

    # 1. Environment
    # env = CustomEnv(...)
    env = None

    # 2. Transform (Optional, defaults to IdentityTransform)
    transform = IdentityTransform()

    # 3. Interpreter
    interpreter = IdentityInterpreter()

    # 4. Output Mode
    output = ScalarOutput(maximize=True)

    # 5. Composition
    return OptimizationEvaluator(
        env=env,
        transform=transform,
        interpreter=interpreter,
        output=output
    )
''')

TRANSFORM_TEMPLATE = string.Template('''from typing import Any
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import BaseTransform
from malthusjax.core.logger import get_logger

logger = get_logger("core.fitness.transform")

@struct.dataclass
class ${name}(BaseTransform[Any]):
    """A custom genotype-to-phenotype transform."""

    def transform(self, genome: Any, state: Any = None) -> Any:
        logger.debug("Applying ${name} transform")
        # TODO: Implement your transformation logic here
        return genome

    # Optional: Override transform_population if the transform
    # must be applied at the population level (e.g. TensorNEAT).
    # def transform_population(self, genes: Any, state: Any = None) -> Any:
    #     logger.debug("Applying population-level transform: ${name}")
    #     return super().transform_population(genes, state)
''')


GENOME_TEMPLATE = string.Template('''from typing import Any, Tuple, Type

import chex
import jax
import jax.numpy as jnp
from flax import struct

from malthusjax.core.base import BaseGenome
from malthusjax.composer import register_genome
from malthusjax.core.logger import get_logger

logger = get_logger("core.genome.${key}")

@register_genome("${key}", override=True)
@struct.dataclass
class ${name}(BaseGenome):
    """A custom genome configuration."""

    values: chex.Array

    @classmethod
    def random_init(cls: Type["${name}"], key: chex.PRNGKey, config: Any) -> "${name}":
        logger.debug("Initializing random genome ${name} with shape (10,)")
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
from malthusjax.core.logger import get_logger

logger = get_logger("core.population")

@struct.dataclass
class ${name}(BasePopulation):
    """A custom population container."""
    # Add your custom fields here
    # Example: custom_metadata: chex.Array = struct.field(default_factory=lambda: jnp.array([]))
    pass
''')

INTERPRETER_TEMPLATE = string.Template('''from typing import Any
import chex
import jax.numpy as jnp
from flax import struct

from malthusjax.core.fitness.composable.base import BaseInterpreter
from malthusjax.core.logger import get_logger

logger = get_logger("core.fitness.interpreter")
# TODO: Import the specific genome type this interpreter consumes, e.g.:
# from malthusjax.core.genome.real_genome import RealGenome

@struct.dataclass
class ${name}(BaseInterpreter[Any]):
    """A custom genome interpreter."""

    @property
    def num_params(self) -> int:
        # TODO: Return the exact number of parameters required by this interpreter.
        # This contract is strictly enforced by MalthusJAX.
        # Return -1 only if the genome length is strictly dictated by the problem.
        return -1

    def apply(self, genome: Any, inputs: chex.Array | None = None) -> chex.Array:
        # TODO: Implement the decoding of the genome and execution over inputs.
        return genome.values
''')

ENVIRONMENT_TEMPLATE = string.Template('''from typing import Any
import chex
import jax.numpy as jnp
from flax import struct

${base_import}
from malthusjax.core.logger import get_logger

logger = get_logger("core.fitness.environment.${key}")

@struct.dataclass
class ${name}(${base_class}):
    """A custom environment."""
${env_body}
''')


# ---------------------------------------------------------------------------
# BackendProvider + Engine scaffolding template
# ---------------------------------------------------------------------------

BACKEND_TEMPLATE = string.Template('''"""${name} backend provider for MalthusJAX.

This module implements the BackendProvider protocol and self-registers
the backend under the key ``"${key}"`` via ``register_backend()``.

Quick-start usage::

    # Import the module so the provider is registered
    import ${module_name}  # noqa: F401

    from malthusjax.composer.backend_registry import get_backends
    from malthusjax.composer.engine_factory_v2 import EngineFactory
    from malthusjax.composer.experiment_config import ExperimentConfig, GenericBackendConfig

    config = ExperimentConfig(
        backend=GenericBackendConfig(name="${key}"),
    )
    engine = EngineFactory.build(config)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp

from malthusjax.composer.backend_registry import register_backend
from malthusjax.composer.engine_protocol import Engine
from malthusjax.composer.strategies.base import BaseStrategy


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ${name}Engine:
    """Concrete engine produced by ${name}Provider.

    Implements the Engine protocol — any object with ``run_once(key, **kwargs)``
    returning a dict of {history, summary, timings}.
    """

    def __init__(
        self,
        # TODO: replace with the real constructor arguments your algorithm needs
        generations: int = 100,
    ) -> None:
        self.generations = generations

    def run_once(self, key: chex.Array, **kwargs: Any) -> Dict[str, Any]:
        """Run one complete evolutionary run and return structured results."""
        t0 = time.perf_counter()
        history: List[Dict[str, Any]] = []

        # TODO: implement your evolutionary loop here
        for gen in range(self.generations):
            best_fitness = 0.0  # replace with real computation
            history.append({"generation": gen, "best_fitness": best_fitness})

        elapsed = time.perf_counter() - t0
        return {
            "history": history,
            "summary": {
                "best_fitness": history[-1]["best_fitness"] if history else None,
                "total_evaluations": self.generations,
            },
            "timings": {
                "total": elapsed,
                "warmup": 0.0,
                "execution": elapsed,
            },
        }


# ---------------------------------------------------------------------------
# BackendProvider
# ---------------------------------------------------------------------------


class ${name}Provider:
    """BackendProvider that builds ${name}Engine instances.

    Satisfies the ``BackendProvider`` runtime-checkable protocol defined in
    ``malthusjax.composer.backend_provider``.
    """

    # ------------------------------------------------------------------ #
    # Identity                                                             #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Canonical registry key — must match the ``register_backend()`` call."""
        return "${key}"

    # ------------------------------------------------------------------ #
    # Strategy                                                             #
    # ------------------------------------------------------------------ #

    def default_strategy(self, **user_kwargs: Any) -> BaseStrategy:
        """Return the default strategy for this backend.

        For simple custom backends, returning a bare ``BaseStrategy()`` is
        sufficient.  Override when you need to carry algorithm hyper-parameters
        in the strategy object.
        """
        # TODO: return a richer strategy dataclass if your backend is
        # parametric, e.g. return MyAlgoStrategy(lr=user_kwargs.get("lr", 1e-3))
        return BaseStrategy()

    def handles_strategy(self, strategy: BaseStrategy) -> bool:
        """Return True when *strategy* was created by this provider.

        Used by the Composer to infer the backend when an explicit strategy
        object is passed without a matching backend name.
        """
        # TODO: replace with isinstance(strategy, MyAlgoStrategy)
        return False

    # ------------------------------------------------------------------ #
    # Evaluator                                                            #
    # ------------------------------------------------------------------ #

    def resolve_evaluator(
        self,
        fitness_spec: Any,
        *,
        maximize: bool = False,
        seed: int = 42,
        num_dims: int = 10,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        **kwargs: Any,
    ) -> Any:
        """Resolve a fitness specification into a callable evaluator.

        Delegate to the shared resolver for standard string/dict specs, or
        return ``fitness_spec`` directly if it is already callable.
        """
        if callable(fitness_spec):
            return fitness_spec

        # For standard MalthusJAX fitness strings (e.g. "sphere:dim=10"),
        # delegate to the shared resolver:
        from malthusjax.composer.backends._evaluator_resolver import resolve_evaluator_base
        return resolve_evaluator_base(
            fitness_spec,
            maximize=maximize,
            seed=seed,
            num_dims=num_dims,
            bounds=bounds,
        )

    # ------------------------------------------------------------------ #
    # Engine construction                                                  #
    # ------------------------------------------------------------------ #

    def build_engine(
        self,
        strategy: BaseStrategy,
        evaluator: Any,
        *,
        pop_size: int = 50,
        generations: int = 100,
        maximize: bool = False,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        history_metrics: Optional[Sequence[str]] = None,
        step_logging: Any = None,
        **kwargs: Any,
    ) -> Engine:
        """Build and return a ${name}Engine instance."""
        # TODO: thread through any algorithm-specific kwargs your engine needs
        return ${name}Engine(generations=generations)

    # ------------------------------------------------------------------ #
    # Population initialisation                                            #
    # ------------------------------------------------------------------ #

    def generate_initial_population(
        self,
        config: Any,
        pop_seed: int,
    ) -> Optional[Any]:
        """Generate a shared initial population (optional).

        Return ``None`` to let the Composer fall back to uniform random
        initialisation.  Override when you need reproducible shared starting
        conditions for ``compare()`` calls.
        """
        # TODO: return a jnp array of shape (pop_size, genome_length) if needed
        return None


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------

_${key}_provider = ${name}Provider()
register_backend("${key}", _${key}_provider)
''')


TEST_BACKEND = string.Template('''"""Compliance tests for ${name}Provider and ${name}Engine."""

from __future__ import annotations

import jax
import pytest

# Import the module so the provider self-registers
from ${import_path}.${module_name} import ${name}Engine, ${name}Provider
from malthusjax.composer.backend_provider import BackendProvider
from malthusjax.composer.backend_registry import get_backends
from malthusjax.composer.engine_protocol import Engine


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class Test${name}ProviderProtocol:
    """Verify that ${name}Provider satisfies the BackendProvider protocol."""

    @pytest.fixture
    def provider(self) -> ${name}Provider:
        return ${name}Provider()

    def test_isinstance_backend_provider(self, provider: ${name}Provider) -> None:
        assert isinstance(provider, BackendProvider), (
            "${name}Provider must satisfy the @runtime_checkable BackendProvider protocol"
        )

    def test_name_property(self, provider: ${name}Provider) -> None:
        assert isinstance(provider.name, str) and provider.name == "${key}"

    def test_default_strategy_returns_base_strategy(self, provider: ${name}Provider) -> None:
        strategy = provider.default_strategy()
        assert strategy is not None

    def test_handles_strategy_returns_bool(self, provider: ${name}Provider) -> None:
        strategy = provider.default_strategy()
        result = provider.handles_strategy(strategy)
        assert isinstance(result, bool)

    def test_resolve_evaluator_returns_object(self, provider: ${name}Provider) -> None:
        # Default: pass a callable through
        fn = lambda x: 0.0
        resolved = provider.resolve_evaluator(fn)
        assert resolved is fn

    def test_generate_initial_population_none_or_array(self, provider: ${name}Provider) -> None:
        result = provider.generate_initial_population(config=None, pop_seed=0)
        # Must be None or a JAX array
        assert result is None or hasattr(result, "shape")


class Test${name}EngineProtocol:
    """Verify that ${name}Engine satisfies the Engine protocol."""

    @pytest.fixture
    def engine(self) -> ${name}Engine:
        return ${name}Engine(generations=5)

    def test_isinstance_engine(self, engine: ${name}Engine) -> None:
        assert isinstance(engine, Engine), (
            "${name}Engine must satisfy the @runtime_checkable Engine protocol"
        )

    def test_run_once_returns_expected_keys(self, engine: ${name}Engine) -> None:
        key = jax.random.PRNGKey(42)
        result = engine.run_once(key)
        assert "history" in result, "RunOutput must contain 'history'"
        assert "summary" in result, "RunOutput must contain 'summary'"
        assert "timings" in result, "RunOutput must contain 'timings'"

    def test_run_once_history_length(self, engine: ${name}Engine) -> None:
        key = jax.random.PRNGKey(0)
        result = engine.run_once(key)
        assert len(result["history"]) == engine.generations


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class Test${name}Registration:
    """Verify that the provider was self-registered correctly."""

    def test_backend_registered(self) -> None:
        backends = get_backends()
        assert "${key}" in backends, (
            f"Backend '${key}' was not found in the registry. "
            "Check that the module-level register_backend() call ran."
        )

    def test_registered_provider_is_correct_type(self) -> None:
        backends = get_backends()
        entry = backends["${key}"]
        provider = entry[0] if isinstance(entry, tuple) else entry
        assert isinstance(provider, ${name}Provider)
''')

# --- TEST TEMPLATES ---

TEST_ADAPTER = string.Template("""import jax
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
""")


TEST_SELECTION = string.Template("""import jax
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
""")

TEST_MUTATION = string.Template("""import jax
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
""")

TEST_CROSSOVER = string.Template("""import jax
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
""")

TEST_EMITTER = string.Template("""import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import EmitterComplianceSuite

class Test${name}(EmitterComplianceSuite):
    @pytest.fixture
    def component(self):
        return ${name}(${dummy_args})
""")

TEST_ENGINE = string.Template("""import jax
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
""")

TEST_EVALUATOR = string.Template('''import jax
import pytest
from ${import_path}.${module_name} import create_${module_name}

def test_${module_name}_creation():
    """Test that the evaluator composition builds successfully."""
    evaluator = create_${module_name}()
    assert evaluator is not None
''')

TEST_TRANSFORM = string.Template("""import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}

class Test${name}:
    @pytest.fixture
    def component(self):
        return ${name}()

    def test_transform(self, component):
        genome = jnp.zeros(10)
        result = component.transform(genome)
        assert result is not None
""")

TEST_GENOME = string.Template("""import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import GenomeComplianceSuite

class Test${name}(GenomeComplianceSuite):
    @pytest.fixture
    def component(self):
        rng = jax.random.PRNGKey(0)
        return ${name}.random_init(rng, config=None)
""")

TEST_POPULATION = string.Template("""import jax
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
""")

TEST_INTERPRETER = string.Template("""import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import InterpreterComplianceSuite

class Test${name}(InterpreterComplianceSuite):
    @pytest.fixture
    def component(self):
        return ${name}()

    @pytest.fixture
    def mock_genome(self):
        from malthusjax.core.genome.real_genome import RealGenome
        return RealGenome(values=jnp.ones(10))

    @pytest.fixture
    def mock_inputs(self):
        return jnp.zeros(5)
""")

TEST_ENVIRONMENT = string.Template("""import jax
import jax.numpy as jnp
import pytest
from ${import_path}.${module_name} import ${name}
from malthusjax.testing.compliance import EnvironmentComplianceSuite

class Test${name}(EnvironmentComplianceSuite):
    @pytest.fixture
    def component(self):
        return ${name}()
""")

DUMMY_ARGS = {
    "selection": "num_selections=5",
    "mutation": "",
    "crossover": "",
    "emitter": "",
    "engine": "",
    "evaluator": "",
    "transform": "",
    "genome": "values=jnp.zeros(10)",
    "population": "",
    "adapter": "",
    "interpreter": "",
    "environment": "",
    "backend": "",
}

TEST_TEMPLATES = {
    "selection": TEST_SELECTION,
    "mutation": TEST_MUTATION,
    "crossover": TEST_CROSSOVER,
    "emitter": TEST_EMITTER,
    "engine": TEST_ENGINE,
    "evaluator": TEST_EVALUATOR,
    "transform": TEST_TRANSFORM,
    "genome": TEST_GENOME,
    "population": TEST_POPULATION,
    "adapter": TEST_ADAPTER,
    "interpreter": TEST_INTERPRETER,
    "environment": TEST_ENVIRONMENT,
    "backend": TEST_BACKEND,
}

TEMPLATES = {
    "selection": SELECTION_TEMPLATE,
    "mutation": MUTATION_TEMPLATE,
    "crossover": CROSSOVER_TEMPLATE,
    "emitter": EMITTER_TEMPLATE,
    "engine": ENGINE_TEMPLATE,
    "evaluator": EVALUATOR_TEMPLATE,
    "transform": TRANSFORM_TEMPLATE,
    "genome": GENOME_TEMPLATE,
    "population": POPULATION_TEMPLATE,
    "adapter": ADAPTER_TEMPLATE,
    "interpreter": INTERPRETER_TEMPLATE,
    "environment": ENVIRONMENT_TEMPLATE,
    "backend": BACKEND_TEMPLATE,
}


def main():
    parser = argparse.ArgumentParser(description="MalthusJAX Component Scaffolder")
    parser.add_argument(
        "--type",
        "-t",
        required=True,
        choices=TEMPLATES.keys(),
        help="Type of component to scaffold",
    )
    parser.add_argument(
        "--name", "-n", required=True, help="Class/Component name (e.g., QuantumMutation)"
    )
    parser.add_argument(
        "--key", "-k", help="Registry key (e.g., quantum). Not required for evaluators."
    )
    parser.add_argument(
        "--impl-dir", "-o", default="plugins", help="Output directory for implementation"
    )
    parser.add_argument(
        "--test-dir", "-to", default="tests/plugins", help="Output directory for tests"
    )
    parser.add_argument(
        "--env-type",
        choices=["opt", "supervised", "rl"],
        default="opt",
        help="Type of environment (only used if type=environment)",
    )

    args = parser.parse_args()

    if args.type != "evaluator" and not args.key:
        print("Error: --key is required for all components except evaluator.")
        return 1

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

    # Prepare specific environment variables
    env_body = ""
    base_class = ""
    base_import = ""
    if args.type == "environment":
        if args.env_type == "rl":
            base_class = "BaseRLEnvironment"
            base_import = "from malthusjax.core.fitness.composable.base import BaseRLEnvironment"
            env_body = """
    def reset(self, key: chex.PRNGKey):
        # TODO: Return initial (obs, state)
        return jnp.zeros(self.obs_dim), None

    def step(self, state: Any, action: chex.Array, key: chex.PRNGKey):
        # TODO: Return (next_obs, next_state, reward, done, info)
        return jnp.zeros(self.obs_dim), state, 0.0, False, {}

    def preprocess_obs(self, obs: Any) -> chex.Array:
        return obs

    def postprocess_action(self, logits: chex.Array, raw_obs: Any = None) -> chex.Array:
        return logits

    @property
    def obs_dim(self) -> int:
        return 4

    @property
    def action_dim(self) -> int:
        return 2
"""
        elif args.env_type == "supervised":
            base_class = "BaseSupervisedEnvironment"
            base_import = (
                "from malthusjax.core.fitness.composable.base import BaseSupervisedEnvironment"
            )
            env_body = """
    # Set data = (X, y) at initialization
    data: Any = struct.field(pytree_node=False, default=None)
"""
        else:  # opt
            base_class = "BaseOptimizationEnvironment"
            base_import = (
                "from malthusjax.core.fitness.composable.base import BaseOptimizationEnvironment"
            )
            env_body = """
    # Add custom static fields (e.g. data for optimization instance)
    # my_data: Any = struct.field(pytree_node=False, default=None)

    def evaluate(self, solution: chex.Array) -> chex.Numeric:
        # TODO: Implement the evaluation of a solution for OptimizationTask.
        return jnp.sum(solution)
"""

    # Render Implementation
    if args.type == "environment":
        impl_content = TEMPLATES[args.type].substitute(
            name=args.name,
            key=args.key,
            base_class=base_class,
            base_import=base_import,
            env_body=env_body,
        )
    elif args.type == "evaluator":
        impl_content = TEMPLATES[args.type].substitute(module_name=module_name)
    elif args.type == "backend":
        impl_content = TEMPLATES[args.type].substitute(
            name=args.name, key=args.key, module_name=module_name
        )
    else:
        impl_content = TEMPLATES[args.type].substitute(name=args.name, key=args.key)

    impl_file.write_text(impl_content)

    import_path = str(out_dir).replace("/", ".")

    # Render Test Boilerplate
    if args.type == "backend":
        test_content = TEST_TEMPLATES[args.type].substitute(
            name=args.name,
            key=args.key,
            module_name=module_name,
            import_path=import_path,
            dummy_args=DUMMY_ARGS[args.type],
        )
    else:
        test_content = TEST_TEMPLATES[args.type].substitute(
            name=args.name,
            module_name=module_name,
            import_path=import_path,
            dummy_args=DUMMY_ARGS[args.type],
        )
    test_file.write_text(test_content)

    print(f"Successfully scaffolded {args.type} '{args.name}'")
    print(f"  Implementation: {impl_file}")
    print(f"  Test:           {test_file}")
    print("  Logging:        Pre-configured via malthusjax.core.logger (get_logger)")
    return 0


if __name__ == "__main__":
    exit(main())
