# `malthusjax.core` — Core Data Structures & Utilities

`malthusjax.core` provides the foundational PyTree representations, genome encodings, struct-of-arrays (SoA) population containers, deterministic PRNG key utilities, and objective function evaluators for **MalthusJAX**.

---

## 1. Core Abstractions (`malthusjax.core.base`)

### `BaseGenome`
Abstract, immutable, JAX PyTree-compatible genome representation. Single-genome methods compose seamlessly with `jax.vmap` to implement population-level operations via the Struct-of-Arrays (SoA) pattern.

**Public API:**
- `random_init(key, config)`: Abstract method to randomly initialize a genome instance.
- `distance(other, metric)`: Computes metric distance between genomes.
- `autocorrect(config)`: Re-enforces domain constraints post-mutation/crossover.
- `clone_buffers()` / `copy()`: Deep-copies JAX array leaves to guarantee safe buffer donation across execution boundaries.
- `create_population(key, config, pop_size)`: Splits `key` and `vmap`s `random_init` to create a batched population genome.

### `BasePopulation[G]`
Struct-of-Arrays (SoA) population container:
- `genes`: Batched genome instance (leading dimension = population size $N$).
- `fitness`: Array of shape `(N,)` storing evaluated objective values.
- `config`: Static genome configuration.
- `info`: Auxiliary dictionary for tracking metrics or Quality-Diversity descriptors.

**Key Methods:**
- `clone_buffers()` / `copy()`: Duplicates underlying JAX array leaves for buffer donation safety.
- `spawn_offspring(new_genes, fitness=None, info=None)`: Constructs an offspring population instance.
- `autocorrect(config)`: Vectorizes `autocorrect` over the population via `jax.vmap`.
- `distance_matrix(metric)`: Computes full $(N, N)$ distance matrix using nested `jax.vmap`.

---

## 2. PRNG & Random Key Management (`malthusjax.core.random`)

Centralizes pseudo-random number generator (PRNG) key construction and handles key compatibility across JAX backends.

**Public API & Enums:**
- `PRNGImpl`: Enum specifying PRNG implementation backends (`THREEFRY`, `PHILOX`, `RBG`, `UNSAFE_RBG`).
- `resolve_prng_impl(name)`: Resolves string aliases or enum instances into valid JAX PRNG backends.
- `create_key(seed, impl=None)`: Constructs typed JAX keys, gracefully falling back if legacy keys are used.
- `validate_key(key, context="")`: Validates PRNG key format at engine initialization boundaries.

---

## 3. Genome Encodings (`malthusjax.core.genome`)

MalthusJAX supports continuous, combinatorial, categorical, and experimental program representations:

| Genome Type | Module | Payload | Key Features & Distance Metrics |
| :--- | :--- | :--- | :--- |
| **Real** | `real_genome.py` | `float32` arrays | Bounds clipping via `autocorrect()`, L2 normalization, Euclidean/Manhattan distance |
| **Binary** | `binary_genome.py` | `{0,1}` integer bits | Bit-to-int conversion (`to_int()`), Hamming/Euclidean distance |
| **Categorical** | `categorical_genome.py` | `int32` category IDs | Permutation validation (`is_permutation()`), swap utilities, Hamming/Euclidean/Manhattan distance |
| **Series** | `series_genome.py` | 2D `(time, features)` arrays | Temporal/feature-axis operators, sequence bounds enforcement |
| **Linear GP** | `linear_genome.py` | `(ops, args)` DAG arrays | Linear Genetic Programming (MEP), DAG causality validation, fast sequence execution |
| **Cartesian GP** | `cartesian_genome.py` | 2D `(nodes, arity)` integers | Cartesian Genetic Programming (CGP), topological levels-back constraints via pure JAX masks |
| **TensorNEAT** | `tensorneat_genome.py` | Graph nodes & connections | Variable-topology neuroevolution networks, compatible with structural mutations and Emitters |

---

## 4. Composable Fitness Evaluators (`malthusjax.core.fitness`)

MalthusJAX decouples objective evaluation using a 4-axis **Composable Evaluator Architecture**:

$$\text{Evaluator} = \text{Shell}(\text{Environment} \times \text{Transform} \times \text{Interpreter} \times \text{Output})$$

This eliminates the coupling between genome decoding and problem environments, allowing any genome representation to be evaluated against any task.

### Composable Task Shells (`composable/evaluators.py`)
- **`OptimizationEvaluator`**: Continuous functions, physics simulations, combinatorial optimization, and analytical benchmarks.
- **`SupervisedEvaluator`**: Supervised learning datasets `(X, y)` computing regression or classification losses.
- **`RLEvaluator`**: Dynamic rollout-based evaluations in reinforcement learning environments (`reset` / `step`).

### Core Composable Primitives (`composable/base.py`)
- **Environment** (`BaseOptimizationEnvironment`, `BaseSupervisedEnvironment`, `BaseRLEnvironment`): Problem logic or dataset definitions.
- **Transform** (`BaseTransform`): Genotype-to-phenotype transformation (e.g., `IdentityTransform`, `TensorNeatTransform`).
- **Interpreter** (`BaseInterpreter`): Decodes genome payloads into callable actions, MLPs, or programs (e.g., `IdentityInterpreter`, `LinearGPInterpreter`).
- **Output** (`ScalarOutput`, `QDOutput`, `MOOutput`): Handles fitness aggregation, optimization direction (`maximize=True/False`), multi-objective metrics, and QD descriptors.

### Out-of-the-Box Evaluators & Benchmarks
- **BBOB-JAX** (`bbobax_evaluator.py`): Pure JAX implementation of the 24 Black-Box Optimization Benchmarks (`BBOBAXEvaluator`).
- **Combinatorial Evaluators** (`binary_evaluators.py`): `BinarySumEvaluator` (OneMax), `KnapsackEvaluator` (0/1 Knapsack with differentiable penalty).
- **Linear GP Evaluators** (`linear_gp_evaluator.py`): Sequence and expression evaluation for Linear Genetic Programming.
- **Third-Party Bridges**: Native adapters for Google Brax, Gymnax, and Instadeep Jumanji.

