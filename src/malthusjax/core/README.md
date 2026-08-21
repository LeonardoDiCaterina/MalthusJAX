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
| **Linear GP** *(Exp)* | `linear_genome.py` | `(ops, args)` DAG instructions | DAG validity enforcement, assembly rendering (`render()`), XLA `lax.switch` interpreter |

---

## 4. Fitness Evaluators (`malthusjax.core.fitness`)

All evaluators adhere to the framework-wide minimization contract (`maximize=False` returns lower-is-better scalar values).

### Evaluator Suite
- **BBOB Suite** (`bbob_evaluator.py`): Wraps standard black-box optimization benchmarking problems via `BBOBEvaluator`.
- **Continuous Evaluators** (`real_evaluators.py`): `SphereEvaluator`, `GriewankEvaluator`, `BoxEvaluator`.
- **Combinatorial Evaluators** (`binary_evaluators.py`): `BinarySumEvaluator` (OneMax), `KnapsackEvaluator` (0/1 Knapsack with linear penalty).
- **Permutation Evaluators** (`tsp_evaluator.py`): `TSPEvaluator` for Traveling Salesperson Problem permutation decoding.
- **Stochastic Evaluators** (`StochasticEvaluator`): Handles PRNG key splitting per individual for noisy/stochastic fitness landscapes.
