# `malthusjax.operators.emitters` — Reference

Scope: `malthusjax.operators.emitters.base`, `malthusjax.operators.emitters.genetic`, `malthusjax.operators.emitters.mixing`, `malthusjax.operators.emitters.qdax_replica`, `malthusjax.operators.emitters.tensorneat_emitter`, `malthusjax.operators.emitters.catalog_factories`. Every claim below is traceable directly to source code and docstrings.

---

## Overview & Architectural Role

In MalthusJAX, **Emitters** (`BaseEmitter`) represent the generative engine of Quality-Diversity (QD), Multi-Objective, and Neuroevolution algorithms. While Level 2 operators (`BaseSelection`, `BaseCrossover`, `BaseMutation`) define pure functional variation transforms on gene arrays, Emitters bridge these atomic operators with population archives and repertoires (Level 3 engines).

```
┌─────────────────────────────────────────────────────────────┐
│                    Repertoire / Archive                     │
│               (e.g., MapElitesRepertoire)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                repertoire.sample() / .select()
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      BaseEmitter                            │
│  ask(state, repertoire, keys) ──> BasePopulation (offspring) │
│  tell(state, repertoire, pop, fitness, descriptors, key)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│     BaseCrossover     │             │     BaseMutation      │
│  (SBX, Blend, etc.)   │             │ (Gaussian, Poly, etc.)│
└───────────────────────┘             └───────────────────────┘
```

Key architectural duties:
- **Parent Sampling**: Queries the repertoire or archive (`repertoire.sample` / `repertoire.select`) to draw parent genotypes.
- **Variation Orchestration**: Composes crossover, mutation, or neural mutations into a vectorized batch operation.
- **Static PRNG Budgeting**: Pre-declares exact random key requirements (`num_keys()`) so the engine's `ResourceMapper` can allocate key slices without dynamic allocations inside `jax.lax.scan`.
- **State Feedback Loop (`ask` / `tell`)**: Exposes an `ask()` method to emit candidate offspring and an optional `tell()` method to update internal emitter parameters (e.g., adaptive step sizes, covariance matrices, novelty metrics).

---

## Base Protocols & Interfaces (`base.py`)

### `EmitterState`
Flax dataclass (`@struct.dataclass`) representing the mutable or carry state of an emitter across generations (e.g. running mutation strengths, covariance matrices, historical statistics).

### `BaseEmitter`
Abstract base class defining the universal emitter contract:

| Method / Property | Signature | Description |
| :--- | :--- | :--- |
| `batch_size` | `@property -> int` | Number of candidate individuals emitted per generation cycle. |
| `num_keys_per_atomic_operation` | `@property -> int` | Number of PRNG subkeys required to produce a single atomic offspring. |
| `num_keys()` | `() -> int` | Total keys needed for the entire batch. Used by `ResourceMapper` for static slice allocation. |
| `set_input_length(length)` | `(int) -> BaseEmitter` | Binds or resizes the emission batch size for static compilation. |
| `init(key, initial_population, params)` | `(chex.Array, BasePopulation, Any) -> Optional[EmitterState]` | Initializes internal emitter state from initial population. |
| `ask(state, repertoire, keys, generation, params)` | `(...) -> Tuple[BasePopulation, Optional[EmitterState]]` | Samples parents from archive and returns candidate offspring population + updated emitter state. |
| `tell(state, repertoire, population, fitnesses, descriptors, key)` | `(...) -> Optional[EmitterState]` | Ingests evaluation metrics and descriptors to adapt emitter hyperparameters. |

### `AtomicEmitter`
Enforces MalthusJAX's **3-Tier Progressive Architecture** for variation emitters:
- **Tier 1 (`_emit_one`)**: Pure atomic function generating a single offspring from parent inputs and subkeys.
- **Tier 2 (`_sample_parents`)**: Sampling from the repertoire and extracting parent gene arrays.
- **Tier 3 (`ask`)**: Vectorized orchestrator that maps `_emit_one` across `batch_size` via `jax.vmap` using pre-budgeted key slices and packages results into a `BasePopulation`.

---

## Built-In Emitters

### 1. Genetic Emitters (`genetic.py`)
Natively wrap standard MalthusJAX Level 2 operators for Quality-Diversity and MAP-Elites:

- **`GeneticMutationEmitter`**: Samples parents from the repertoire and applies a `BaseMutation` operator (e.g., `GaussianMutation`, `PolynomialMutation`). Forwards generational step counters to support scheduled mutation annealing (e.g., `LINEAR_DECAY`, `COSINE_ANNEAL`).
- **`GeneticCrossoverEmitter`**: Samples pairs of parents from the archive and executes a `BaseCrossover` operator (e.g., `SBXCrossover`, `UniformCrossover`).
- **`GeneticMixingEmitter`**: Dual-pipeline genetic emitter executing both crossover and mutation, with configurable operator ratios.

### 2. Compositional Emitters (`mixing.py`)
- **`MixingEmitter`**: Manages two distinct sub-emitters (`emitter_a` and `emitter_b`) in parallel. Splits generational quota between both emitters, concatenates their produced offspring populations, and tracks composite `MixingEmitterState(state_a, state_b)`.

### 3. Ecosystem & Framework Emitters
- **`QDAXReplicaEmitter` (`qdax_replica.py`)**: Pure-JAX reimplementation of QDAX variation emitters for MAP-Elites, providing byte-exact parity with external QDAX benchmarks without requiring external C++ dependencies.
- **`TensorNeatEmitter` (`tensorneat_emitter.py`, `tensorneat_variants.py`)**: Neuroevolution emitter designed for variable-topology neural networks, orchestrating structural mutation (adding nodes, adding connections) and weight mutations compatible with TensorNEAT.

---

## Catalog Registration & Scaffolding

### Declarative Registration
Emitters are registered into the central Composer registry via `@register_emitter`:

```python
from flax import struct
from malthusjax.composer import register_emitter
from malthusjax.operators.emitters.base import BaseEmitter

@register_emitter("custom_gaussian_emitter", override=True)
@struct.dataclass
class CustomGaussianEmitter(BaseEmitter):
    ...
```

### CLI Scaffolding
Scaffold a production-ready emitter with boilerplate implementation and compliance test suite in one command:

```bash
python scripts/scaffold.py -t emitter -n CovarianceEmitter -k covariance_emitter -o src/malthusjax/plugins -to tests/plugins
```

This generates:
1. `src/malthusjax/plugins/covariance_emitter.py`: Implements `BaseEmitter` dataclass with `ask()`, `init()`, and `set_input_length()`.
2. `tests/plugins/test_covariance_emitter.py`: Inherits from `EmitterComplianceSuite` to automatically verify JIT compliance, shape contracts, and PRNG key budgets.

---

## Compliance & Verification (`EmitterComplianceSuite`)

Custom emitters are validated using `malthusjax.testing.compliance.EmitterComplianceSuite`:

```python
import pytest
from malthusjax.testing.compliance import EmitterComplianceSuite
from my_module import MyEmitter

class TestMyEmitter(EmitterComplianceSuite):
    @pytest.fixture
    def component(self):
        return MyEmitter()
```

Automated assertions include:
- `test_is_dataclass`: Confirms the class is a `@flax.struct.dataclass`.
- `test_inheritance`: Verifies inheritance from `BaseEmitter`.
- `test_batch_size_contract`: Ensures `batch_size > 0`.
- `test_num_keys_contract`: Ensures `num_keys() >= 0` for `ResourceMapper`.
- `test_has_registry_metadata`: Confirms `@register_emitter` decorator attachment.
- `test_ask_execution`: Executes `ask()` with PRNG keys and verifies returned population length equals `batch_size`.
