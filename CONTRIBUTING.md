# Contributing to MalthusJAX

First of all, thank you for your interest in contributing to MalthusJAX! 

MalthusJAX is designed to be highly modular and extensible. Our overarching vision is to construct a comprehensive "rack" of **Evolutionary Engines** and **Genetic Operators**.

We welcome contributions of all kinds, but we are particularly interested in:
- **Evosax Matches**: Expanding our engine registry to achieve feature parity with all strategies found in the `evosax` library.
- **Novel Operators**: New implementations of selection, mutation, and crossover mechanisms.
- **Domain-Specific Extensions**: Specialized genome types and operators tailored for advanced use-cases (e.g., protein folding, language model optimization, compiler pass sequencing).

## The Golden Rules

MalthusJAX is strictly built on top of JAX and Flax. To ensure high performance, auto-vectorization (`vmap`), and Just-In-Time (`jit`) compilation, all core components must adhere to the following rules:

1. **JAX Compliance**: Your code must be compatible with `jax.jit` and `jax.vmap`. Avoid Python control flow (`if`, `for`) dependent on dynamic array values. Use JAX control flow (`jax.lax.cond`, `jax.lax.scan`) instead.
2. **Immutability and Statelessness**: All component state must be explicitly managed and passed through functions. Components themselves are stateless, immutable data structures.
3. **Flax Structs**: All components (Engines, Operators, Fitness Evaluators, Genome Configs) **must** be implemented as `@flax.struct.dataclass`. 

## Generating Boilerplate with the Scaffolding CLI

To drastically reduce friction, MalthusJAX provides a built-in Scaffolding CLI to instantly generate boilerplate code and tests for any component.

To use the CLI, simply run the `scaffold` Makefile target from the repository root:

```bash
# Example: Creating a new mutation operator
make scaffold ARGS="--type mutation --name QuantumMutation --key quantum"

# Example: Creating a new evolutionary engine
make scaffold ARGS="--type engine --name CMAESEngine --key cmaes"
```

### What this does:
1. Generates the component implementation in the `plugins/` directory.
2. Generates a dummy test suite in `tests/plugins/`.
3. Auto-generates the correct inheritance (`BaseMutation`, `AbstractEngine`, etc.).
4. Injects the appropriate `@register_*` decorator to tie the component into the global MalthusJAX registry.

Once generated, simply fill in the `TODO` sections with your specific JAX logic!

## The Component Registry

MalthusJAX relies on a dynamic global registry system. By decorating your class with the appropriate register decorator (e.g., `@register_mutation("my_key")`), the `composer` can dynamically discover and validate it. 

When contributing an operator, make sure to specify its compatibility correctly using metadata parameters. For example:

```python
from flax import struct
from malthusjax.operators.base import BaseMutation
from malthusjax.composer import register_mutation

@register_mutation("custom_mut", compatible_genomes=["real", "continuous"])
@struct.dataclass
class CustomMutation(BaseMutation):
    ...
```

## Pull Request Guidelines

1. **Test Coverage**: All new operators and engines must be accompanied by a unit test proving they can instantiate and execute without JAX tracer errors. (The scaffolding CLI gives you a head start here!).
2. **Metadata**: Ensure you provide accurate metadata in the `@register_*` decorators so the Composer validation framework can correctly check pipeline compatibility.
3. **Run the Checks**: Before submitting, ensure your code passes linting, formatting, type-checking, and tests by running:
   ```bash
   make check-all
   ```

We look forward to seeing what you build!
