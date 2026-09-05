# Evaluator Architecture: Composable Fitness Evaluation

> **Status**: Design Proposal — `feature/evaluator-task-refactor` branch  
> **Authors**: MalthusJAX Core Team  
> **Related branches**: `main` (current monolithic evaluators), `feature/evaluator-task-refactor` (this refactor)

---

## 1. Motivation

### The Problem with the Current Architecture

Every evaluator in the current MalthusJAX codebase secretly bundles two completely independent concerns into a single class:

**Concern A — Interpretation**: How do we turn a raw genome (a JAX array) into something that can *act* in an environment?

**Concern B — Environment/Data**: What environment, dataset, or benchmark function does the interpreted genome interact with to produce a scalar fitness?

This tight coupling causes a **combinatorial explosion**. Every time a new dataset is added, a new evaluator must be written for every genome type that might evaluate on it:

| | SKLearn Dataset | Equation Generator | Custom Dataset |
|---|---|---|---|
| `LinearGPGenome` | `SklearnMEPEvaluator` | `EquationMEPEvaluator` | `CustomMEPEvaluator` |
| `CartesianGPGenome` | `SklearnCGPEvaluator` | `EquationCGPEvaluator` | `CustomCGPEvaluator` |
| `TensorNEATGenome` | **impossible** (bridge hack needed) | **impossible** | **impossible** |

The `TensorNEATGenome` row is completely broken: because `SklearnMEPEvaluator.evaluate()` expects a MEP program string, it cannot be used to evaluate a TensorNEAT neural network. A manual, one-off bridge class (`TensorNEATSupervisedProblem`) was required.

Furthermore, QD (Quality Diversity) and Multi-Objective evaluators are currently handled by subclassing, scattering descriptor computation logic throughout the evaluator hierarchy when it should be orthogonal to it.

### The Goal

Replace the monolithic evaluator with a **three-axis decomposition** that achieves:

1. **Zero combinatorial explosion**: write each component once, mix freely.
2. **TensorNEAT and all future adapters work natively** with any environment.
3. **QD and MO** are orthogonal extensions, not subclasses.
4. **JAX-friendly**: all components are stateless, JIT-compilable, and shape-static.

---

## 2. Core Vocabulary

The new architecture introduces four distinct concepts:

### 2.1 Task Type (Abstract Category)

The abstract *kind* of learning problem. Determines the **evaluation loop structure**.

| Task Type | Pattern | Output Contract |
|---|---|---|
| `OptimizationTask` | `f(genome, instance_data) → scalar` | Direct evaluation of genome as solution against a problem instance |
| `SupervisedTask` | `predict(genome, X) → y_hat; loss(y_hat, y)` | Prediction over static dataset |
| `ReinforcementTask` | `reset(); step() × T; accumulate reward` | Episode rollout |
| `UnsupervisedTask` | TBD | TBD |

> **Key insight for `OptimizationTask`**: The problem instance data *defines* f(x). For BBOB the instance is analytically described (no data needed). For TSP the instance is the city coordinates matrix. For Knapsack it is the item weights, values, and capacity. In all cases the genome *is* the candidate solution, evaluated directly — there is no MLP inference or program execution in between.

The Task Type is **fixed by the environment** — a `SklearnEnv` is always a `SupervisedTask`. A `GymnaxEnv` is always a `ReinforcementTask`. Users never declare the Task Type explicitly; it is implied by their Environment choice.

### 2.2 Environment (Concrete Implementation)

The *specific* environment, dataset, or benchmark function. Belongs to exactly one Task Type.

```
SupervisedTask environments:
├── SklearnEnv(dataset="breast_cancer")         # (X, y) from scikit-learn
├── EquationEnv(equation="feynman_01")           # (X, y) generated from equation
└── CustomDatasetEnv(X=my_X, y=my_y)            # user-provided arrays

ReinforcementTask environments:
├── GymnaxEnv(env_name="CartPole-v1")            # JAX-native RL (gymnax)
├── JumanjiEnv(env_name="RubiksCube-v0")         # JAX-native combinatorial RL (jumanji)
└── BraxEnv(env_name="ant")                      # physics-based RL (brax)

OptimizationTask environments:
├── BBOBEnv(fn_name="sphere", num_dims=10)       # analytic black-box (no data needed)
├── BBOBAxEnv(fn_name="rastrigin", num_dims=10)  # analytic black-box (bbobax)
├── TSPEnv(cities=city_coords)                   # TSP instance — data defines f(tour)
├── KnapsackEnv(weights=w, values=v, cap=C)      # Knapsack instance — data defines f(selection)
└── GraphColoringEnv(adjacency=adj_matrix)       # graph problem instance
```

The Environment exposes a unified interface per Task Type so that the Evaluator loop never needs to know which library it is talking to. For example, all `ReinforcementTask` environments expose:

```python
class BaseRLEnvironment:
    def reset(self, key) -> (obs, state): ...
    def step(self, state, action, key) -> (obs, state, reward, done): ...
    def preprocess_obs(self, obs) -> flat_array: ...
    def postprocess_action(self, logits) -> action: ...
    
    @property
    def obs_dim(self) -> int: ...
    
    @property
    def action_dim(self) -> int: ...
```

### 2.3 Interpreter (Genome Decoder)

The bridge that decodes a **genome** into a **callable** that can interact with an environment. It is:

- **Stateless**: all configuration (input/output dimensions, architecture) is compiled in at construction time.
- **Genome-specific**: typed to a specific genome type.
- **Task-agnostic**: its output is always `apply(genome, inputs) → outputs`.

The core interface:

```python
class BaseInterpreter(Generic[G]):
    """Decodes a genome into a callable over inputs.
    
    The key method signature is:
        apply(genome: G, inputs: Array) -> Array
    """
    def apply(self, genome: G, inputs: Array) -> Array:
        raise NotImplementedError
        
    @property
    def num_params(self) -> int:
        """Total parameter count — defines the required genome length."""
        raise NotImplementedError
```

#### Origin of Interpreters

There are two origins, depending on the framework:

**For native MalthusJAX genomes** — the Interpreter is declared explicitly by the user:

| Interpreter | Genome | Description |
|---|---|---|
| `IdentityInterpreter` | `RealGenome`, `BinaryGenome` | Genome IS the solution — passes raw values directly to the environment (BBOB, TSP, Knapsack) |
| `MLPInterpreter(input_dim, output_dim, hidden)` | `RealGenome` | Unflatten flat array → MLP weights → apply to inputs (RL/SL) |
| `LinearGPInterpreter` | `LinearGenome` | Execute MEP program over inputs (SL) |
| `CartesianGPInterpreter` | `CartesianGenome` | Evaluate CGP graph over inputs (SL) |

**For adapter frameworks** (TensorNEAT, evosax) — the Interpreter is provided automatically by the adapter. The user never constructs it. The adapter's internal representation compiles the genome into a callable:

| Interpreter | Genome | Provided by |
|---|---|---|
| `TensorNEATInterpreter` | `TensorNEATGenome` | `TensorNEATAdapter` |
| `EvosaxInterpreter` | `EvosaxGenome` | `EvosaxAdapter` |

#### The Self-Consistency Constraint

Because `MLPInterpreter` specifies `input_dim`, `output_dim`, and `hidden`, it knows its total parameter count:

```python
interp = MLPInterpreter(input_dim=4, output_dim=2, hidden=[64, 64])
# interp.num_params == 4*64 + 64 + 64*2 + 2 == 450

# The genome MUST have exactly this many values:
genome = RealGenomeSpec(length=interp.num_params)
```

This constraint is **checkable at construction time**, not at runtime.

### 2.4 Output Mode (Engine Contract)

Defines what the Evaluator **returns** to the engine. This is **orthogonal** to Task Type, Environment, and Interpreter — you can apply any Output Mode to any combination of the other three.

| Output Mode | Returns | Used by |
|---|---|---|
| `ScalarOutput` | `fitness: Array[pop_size]` | Standard GA, ES |
| `QDOutput(descriptor_fn)` | `(fitness, descriptors)` | MAP-Elites, CVT-MAP-Elites |
| `MOOutput` | `fitness_vector: Array[pop_size, n_obj]` | NSGA-II, MOEA/D |

#### QD Descriptors are Injectable

The `QDOutput` takes a `DescriptorFn` as a separate, injectable object. The Evaluator itself knows **nothing** about descriptors:

```python
class BaseDescriptorFn:
    """Computes behavior descriptors from a genome and evaluation result."""
    def compute(self, genome, inputs, outputs) -> Array: ...

class DimensionActivityDescriptor(BaseDescriptorFn):
    """Describes which input dimensions dominate the genome's behaviour."""
    
class EpisodeLengthDescriptor(BaseDescriptorFn):
    """Describes average episode length for RL rollouts."""
```

The Evaluator produces the raw result; `DescriptorFn` computes the behavior characterization from it. These two concerns are fully decoupled.

---

## 3. The Composition Formula

```
Problem  = TaskType × Environment × Interpreter
Output   = ScalarOutput | QDOutput(DescriptorFn) | MOOutput
Evaluator = Problem × Output
```

The Evaluator is now just the **orchestrator** — it runs the evaluation loop defined by the Task Type, using the Environment for data/dynamics and the Interpreter to decode genomes. It then packages results according to the Output Mode.

### 3.1 Usage Examples

#### Standard GA on BBOB (continuous optimization)

```python
env = BBOBEnv(fn_name="sphere", num_dims=10)
interp = IdentityInterpreter(dim=10)  # genome IS the solution vector
output = ScalarOutput()

evaluator = OptimizationEvaluator(env=env, interpreter=interp, output=output)
```

#### Standard GA on TSP (combinatorial optimization)

```python
env = TSPEnv(cities=city_coordinates)  # problem instance data defines f(tour)
interp = IdentityInterpreter(dim=n_cities)  # genome IS the tour encoding
output = ScalarOutput()

# Same evaluator class — different environment, zero new code
evaluator = OptimizationEvaluator(env=env, interpreter=interp, output=output)
```

#### MalthusJAX MLP on SKLearn Regression

```python
env = SklearnEnv(dataset="california_housing")  # provides X, y
interp = MLPInterpreter(input_dim=8, output_dim=1, hidden=[64, 64])  
# interp.num_params == 8*64 + 64 + 64*1 + 1 == 641
output = ScalarOutput(loss_fn="mse")

evaluator = SupervisedEvaluator(env=env, interpreter=interp, output=output)
```

#### TensorNEAT on SKLearn Classification (the previously impossible case)

```python
env = SklearnEnv(dataset="breast_cancer")  # same env as above!
interp = TensorNEATInterpreter(algorithm=neat_algo, state=neat_state)  # adapter provides this
output = ScalarOutput(loss_fn="bce")

evaluator = SupervisedEvaluator(env=env, interpreter=interp, output=output)
# TensorNEAT works on sklearn with zero bridge code!
```

#### MLP on Gymnax (RL)

```python
env = GymnaxEnv(env_name="CartPole-v1")  # obs_dim=4, action_dim=2
interp = MLPInterpreter(input_dim=4, output_dim=2, hidden=[64, 64])
output = ScalarOutput()

evaluator = RLEvaluator(env=env, interpreter=interp, output=output)
```

#### TensorNEAT on Jumanji (RL — the ultimate mix-and-match)

```python
env = JumanjiEnv(env_name="TSP-v1")  # different library, same interface!
interp = TensorNEATInterpreter(algorithm=neat_algo, state=neat_state)
output = ScalarOutput()

evaluator = RLEvaluator(env=env, interpreter=interp, output=output)
# TensorNEAT on Jumanji: zero new code needed
```

#### MAP-Elites QD on Gymnax

```python
env = GymnaxEnv(env_name="Ant-v1")
interp = MLPInterpreter(input_dim=27, output_dim=8, hidden=[256, 256])
descriptor_fn = EpisodeFeatureDescriptor(features=["avg_velocity", "episode_length"])
output = QDOutput(descriptor_fn=descriptor_fn)

evaluator = RLEvaluator(env=env, interpreter=interp, output=output)
# MAP-Elites can consume this evaluator directly
```

---

## 4. The Mix-and-Match Matrix

The power of this decomposition is that any valid combination just works:

| Interpreter | `SklearnEnv` | `GymnaxEnv` | `JumanjiEnv` | `BBOBEnv` | `TSPEnv` | `KnapsackEnv` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `IdentityInterpreter` | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `MLPInterpreter` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `LinearGPInterpreter` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `CartesianGPInterpreter` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `TensorNEATInterpreter` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

> **Why ❌ for `IdentityInterpreter` + RL/SL?** RL and SL require the genome to produce a *callable* (a policy or predictor). `IdentityInterpreter` outputs a raw vector — it is meaningless as a policy. Only `OptimizationTask` environments accept raw vectors directly.
>
> **Why ❌ for `MLPInterpreter` + Optimization?** Optimization tasks do not have an input → output inference step. There is no `obs` to feed into the MLP. The solution *is* the genome.

Any `✅` combination produces a valid `Evaluator` with **zero new code**.

---

## 5. Backward Compatibility

The existing evaluator classes (`BBOBEvaluator`, `LinearGPEvaluator`, `SklearnMEPEvaluator`, etc.) will continue to function as convenience wrappers. Internally they will be refactored to compose the new primitives, but their external interface stays unchanged.

```python
# Old API — still works
evaluator = SklearnMEPEvaluator(dataset="make_regression", n_samples=50, n_features=5)

# New API — full control
env = SklearnEnv(dataset="make_regression", n_samples=50, n_features=5)
interp = LinearGPInterpreter(num_inputs=5, length=100)
evaluator = SupervisedEvaluator(env=env, interpreter=interp, output=ScalarOutput(loss_fn="mse"))
```

---

## 6. Implementation Roadmap

### Phase 1: Core Interfaces
- `src/malthusjax/core/fitness/environments/base.py` — `BaseEnvironment`, `BaseRLEnvironment`, `BaseSupervisedEnvironment`
- `src/malthusjax/core/fitness/interpreters/base.py` — `BaseInterpreter`
- `src/malthusjax/core/fitness/output/base.py` — `BaseOutputMode`, `ScalarOutput`, `QDOutput`, `MOOutput`

### Phase 2: Concrete Environments
- `src/malthusjax/core/fitness/environments/sklearn_env.py` — `SklearnEnv`
- `src/malthusjax/core/fitness/environments/gymnax_env.py` — `GymnaxEnv`
- `src/malthusjax/core/fitness/environments/jumanji_env.py` — `JumanjiEnv`
- `src/malthusjax/core/fitness/environments/bbob_env.py` — `BBOBEnv`, `BBOBAxEnv`

### Phase 3: Concrete Interpreters
- `src/malthusjax/core/fitness/interpreters/identity.py` — `IdentityInterpreter`
- `src/malthusjax/core/fitness/interpreters/mlp.py` — `MLPInterpreter`
- `src/malthusjax/core/fitness/interpreters/linear_gp.py` — `LinearGPInterpreter`
- `src/malthusjax/core/fitness/interpreters/cartesian_gp.py` — `CartesianGPInterpreter`

### Phase 4: Evaluator Loops
- `src/malthusjax/core/fitness/evaluators/supervised.py` — `SupervisedEvaluator`
- `src/malthusjax/core/fitness/evaluators/rl.py` — `RLEvaluator`
- `src/malthusjax/core/fitness/evaluators/optimization.py` — `OptimizationEvaluator`
- `src/malthusjax/core/fitness/environments/tsp_env.py` — `TSPEnv`
- `src/malthusjax/core/fitness/environments/knapsack_env.py` — `KnapsackEnv`

### Phase 5: Adapter Integration
- Update `TensorNEATAdapter` to expose `TensorNEATInterpreter`
- Update `EvosaxAdapter` to expose `EvosaxInterpreter`

### Phase 6: Composer Integration
- Update `Composer` to accept the three-axis decomposition as first-class parameters
- Maintain backward-compatible string parsing for legacy `fitness="sklearn_mep"` syntax

### Phase 7: Descriptor Functions
- `src/malthusjax/core/fitness/descriptors/base.py` — `BaseDescriptorFn`
- Migrate existing QD evaluator descriptor logic into injectable `DescriptorFn` objects

---

## 7. Design Decisions

### 7.1 Composer API

The current Composer TOML uses flat string keys:
```toml
fitness = "sklearn_mep:dataset=make_regression,n_samples=50,n_features=5"
```

This works but becomes **unreadable** and **unextensible** for the three-axis decomposition. We need to express `env`, `interpreter`, and `output` cleanly. Two options were evaluated:

---

**Option A: Fully Decomposed (breaking change)**

```toml
[experiment.shared.fitness]
env        = "sklearn"
env.dataset = "make_regression"
env.n_samples = 50
env.n_features = 5

interpreter        = "mlp"
interpreter.input_dim  = 5
interpreter.output_dim = 1
interpreter.hidden = [64, 64]

output     = "scalar"
output.loss_fn = "mse"
```

Python API equivalent:
```python
Composer(
    env=SklearnEnv(dataset="make_regression", n_samples=50, n_features=5),
    interpreter=MLPInterpreter(input_dim=5, output_dim=1, hidden=[64, 64]),
    output=ScalarOutput(loss_fn="mse"),
)
```

✅ Clean. Each concern is explicit and independently configurable.  
✅ Scales to any combination without new string parsers.  
❌ Breaks all existing configs.

---

**Option B: Backward-Compatible TOML Section**

Old configs keep working. New configs can use a structured `[fitness]` subsection:

```toml
# Old style — still works
fitness = "sklearn_mep:dataset=make_regression,n_samples=50,n_features=5"

# OR: new structured style for the new architecture
[pipelines.tensorneat.fitness]
env            = "sklearn"
env.dataset    = "breast_cancer"
interpreter    = "tensorneat"     # adapter provides this automatically
output         = "scalar"
output.loss_fn = "bce"
```

The Composer detects which format is used:
- String `fitness = "..."` → legacy string parser (existing behavior)
- Table `[fitness]` → new three-axis parser

✅ Zero migration cost for existing configs.  
✅ New configs are clean and structured.  
✅ Both syntaxes can coexist in the same TOML (pipeline-level override still works).  
❌ Two parsers to maintain during transition.

---

**Decision**: **Option B** — structured `[fitness]` section with backward compatibility.

The `fitness = "sklearn_mep"` string syntax represents the *old* bundled world and continues to work forever as a convenience alias. New code and new configs use `[fitness]` with explicit keys. When the `[fitness]` table is present, it takes precedence over the string key.

---

### 7.2 Adapter Interpreter Initialization

For adapter frameworks (TensorNEAT), there is a dependency order:
1. The algorithm must be initialized first: `state = algorithm.init(key)` — this compiles the network graph, sets up jit-traced operations, etc.
2. Only then can `TensorNEATInterpreter(algorithm=algo, state=state)` be created.

**Option A: Adapter exposes interpreter as a property**

The adapter initializes everything and makes the interpreter available:
```python
adapter = TensorNEATAdapter(algorithm=neat)
adapter.initialize(key)  # runs algorithm.init() internally
interpreter = adapter.interpreter  # TensorNEATInterpreter(algo, state) ready to use
```

✅ User never manually manages TensorNEAT state.  
✅ The adapter is the single authority on its own initialization.  
✅ Consistent with how adapters are used elsewhere in MalthusJAX.  

**Option B: User constructs interpreter directly**

```python
state = neat.init(key)
interpreter = TensorNEATInterpreter(algorithm=neat, state=state)
```

✅ Explicit and transparent.  
❌ Requires user to manage TensorNEAT internals.  
❌ Breaks encapsulation — the adapter abstraction exists precisely to hide this.

**Decision**: **Option A** — the adapter provides the interpreter via a property. The Composer calls `adapter.interpreter` automatically when `interpreter = "tensorneat"` is declared. For native MalthusJAX interpreters (MLPInterpreter, LinearGPInterpreter), the user always constructs them directly.

---

### 7.3 Genome Spec Derivation

**Decision**: **Explicit declaration with loud validation**.

The user always declares the genome length explicitly:
```python
interp = MLPInterpreter(input_dim=4, output_dim=2, hidden=[64, 64])
# interp.num_params == 450

genome_spec = RealGenomeSpec(length=450)   # user declares explicitly
```

At Composer build time, a validation step checks:
```python
if genome_spec.length != interpreter.num_params:
    raise ConfigurationError(
        f"Genome length {genome_spec.length} does not match "
        f"MLPInterpreter.num_params={interpreter.num_params}. "
        f"Set genome_length = {interpreter.num_params} in your config."
    )
```

This is intentionally **loud**. Silent auto-configuration would hide bugs where the wrong genome size is accidentally used. By requiring explicit declaration, mismatches are caught before any JAX tracing begins, with a clear error message pointing to the exact fix.

For TOML configs:
```toml
genome_length = 450   # must match interpreter.num_params exactly — Composer validates this

[pipelines.mlp_cartpole.fitness]
interpreter        = "mlp"
interpreter.input_dim  = 4
interpreter.output_dim = 2
interpreter.hidden = [64, 64]
# Composer will raise ConfigurationError if genome_length != 450
```

