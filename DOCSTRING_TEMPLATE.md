# NumPy Docstring Format Guide for MalthusJAX

Quick reference for applying NumPy-style docstrings to MalthusJAX codebase.

---

## Template 1: Class Docstring

```python
class MyOperator(BaseOperator):
    """Short summary in imperative mood, no period.
    
    Longer description explaining the algorithm and when to use it.
    Reference papers, JAX-specific behavior, complexity, etc.
    
    Parameters
    ----------
    param1 : type
        Description of param1. For string specs, show format:
        ``"operator_name:key1=val1,key2=val2"``.
    param2 : int, optional
        Description. Default is 10.
    
    Attributes
    ----------
    param1 : type
        The param1 value.
    param2 : int
        The param2 value.
    
    Raises
    ------
    ValueError
        If param1 is invalid format.
    
    Notes
    -----
    JAX considerations:
    
    - JIT-compilable: Yes
    - PRNG keys consumed: 1 per call (shape `(2,)`)
    - Pytree-safe: Yes
    
    References
    ----------
    .. [1] Smith & Jones (2020). "Algorithm Name". Journal.
    
    Examples
    --------
    >>> op = MyOperator(param1="sphere:dim=10")
    >>> keys = jax.random.split(jax.random.PRNGKey(0), op.num_keys(pop_size))
    >>> result = op(keys, population, config)
    """
    
    def __init__(self, param1: str, param2: int = 10):
        ...
```

---

## Template 2: Method Docstring (Operator Call)

```python
def __call__(
    self,
    all_keys: jax.Array,
    population: BasePopulation,
    config: BaseGenomeConfig,
    generation: int = 0,
) -> BasePopulation:
    """Apply operator to population and return offspring.
    
    Detailed explanation of what happens in this transformation.
    Reference the algorithm if not obvious from class docstring.
    
    Parameters
    ----------
    all_keys : jax.Array
        PRNG keys with shape `(num_pairs, num_offspring, num_keys_per_op, 2)`.
        Generate via: ``jax.random.split(base_key, self.num_keys(pop_size))``.
    population : BasePopulation
        Input population. Attributes: genes (pytree), fitness (1D array).
    config : BaseGenomeConfig
        Genome configuration (bounds, shape, etc.).
    generation : int, optional
        Current generation number (for scheduling). Default is 0.
    
    Returns
    -------
    BasePopulation
        Offspring population with same structure as input. Fitness values
        are set to NaN (caller must evaluate).
    
    Notes
    -----
    - Consumes one PRNG key (shape `(2,)`). Multiple keys allocated
      internally for vectorized operations.
    - JAX-compatible: Fully JIT-compilable.
    - Pytree structure: Output is standard pytree, safe for use in
      jitted functions and transformations.
    
    Examples
    --------
    >>> config = RealGenomeConfig(shape=(10,), bounds=(-5, 5))
    >>> pop = RealPopulation.init_random(key, config, size=32)
    >>> op = MyOperator()
    >>> key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    >>> keys = jax.random.split(key1, op.num_keys((16,)))
    >>> offspring = op(keys, pop, config)
    >>> print(offspring.values.shape)  # (32, 10)
    """
```

---

## Template 3: Function Docstring (Factory/Helper)

```python
def create_operator_from_spec(spec: str) -> Operator:
    """Create an operator instance from specification string.
    
    Specification strings provide a concise way to configure operators
    without explicitly instantiating classes. This is useful for
    configuration files and hyperparameter sweeps.
    
    Parameters
    ----------
    spec : str
        Specification string with format: ``"operator_name:param1=val1,param2=val2"``.
        
        Valid operator names and parameters:
        
        - ``"gaussian:rate=0.1,strength=1.0"`` — Gaussian mutation
        - ``"bitflip:rate=0.01"`` — Bit-flip mutation
        - ``"sbx:eta=15.0,num_offspring=2"`` — Simulated binary crossover
        - ``"tournament:tournament_size=3,num_selections=32"`` — Tournament selection
    
    Returns
    -------
    Operator
        Configured operator instance ready for use.
    
    Raises
    ------
    ValueError
        If spec string is not in recognized format or contains invalid parameters.
    KeyError
        If operator name is not registered.
    
    Examples
    --------
    >>> op1 = create_operator_from_spec("gaussian:rate=0.1,strength=0.5")
    >>> op2 = create_operator_from_spec("tournament:tournament_size=5")
    >>> print(type(op1).__name__)
    GaussianMutation
    """
```

---

## Template 4: Dataclass with pytree_node=False

```python
@struct.dataclass
class MyConfig:
    """Configuration for MyOperator.
    
    Stores operator hyperparameters. Can be instantiated from dictionaries
    or spec strings. All instances are frozen (immutable in JAX terms).
    
    Parameters
    ----------
    name : str
        Human-readable name for this configuration. 
        NON-PYTREE FIELD: This field is not included in JAX transformations.
        Changing it between JIT calls will trigger recompilation.
    param1 : float
        Some hyperparameter controlling behavior.
    param2 : int, optional
        Another hyperparameter. Default is 10.
    
    Attributes
    ----------
    name : str
        The configuration name.
    param1 : float
        The first hyperparameter value.
    param2 : int
        The second hyperparameter value.
    
    Notes
    -----
    JAX compatibility:
    
    - Pytree fields: param1, param2
    - Non-pytree fields: name (for logging/tracking only)
    - Frozen: All fields are immutable
    - Can be used inside jit() functions safely
    
    Examples
    --------
    >>> config = MyConfig(name="experiment_1", param1=0.5, param2=20)
    >>> print(config.name)
    experiment_1
    """
    
    name: str = struct.field(metadata=dict(pytree_node=False))
    param1: float
    param2: int = 10
```

---

## Template 5: Array Parameter with Shape

```python
def vectorized_fitness(
    genes: jax.Array,
    config: GenomeConfig,
) -> jax.Array:
    """Evaluate fitness for a batch of genomes.
    
    Vectorized evaluation using JAX operations. Can be used with vmap
    for efficient batched evaluation.
    
    Parameters
    ----------
    genes : jax.Array
        Genes to evaluate with shape `(pop_size, genome_dim)`.
        Values should be within config.bounds.
    config : GenomeConfig
        Configuration specifying bounds and interpretation of genes.
    
    Returns
    -------
    jax.Array
        Fitness values with shape `(pop_size,)`. Higher values are better
        (maximization problem). NaN indicates invalid genes.
    
    Notes
    -----
    - Input genes are NOT modified (pure function).
    - JAX-compatible: Fully vectorized, can use with vmap/grad/jit.
    - Bounds checking: No automatic clipping; out-of-bounds genes
      return NaN.
    
    Examples
    --------
    >>> genes = jax.random.uniform(key, shape=(32, 10), minval=-5, maxval=5)
    >>> fitness = vectorized_fitness(genes, config)
    >>> print(fitness.shape)
    (32,)
    """
```

---

## Checklist Before Submitting Docstring PR

For each file refactored:

- [ ] All public classes have docstrings (summaries + sections)
- [ ] All public methods have docstrings
- [ ] All public functions have docstrings
- [ ] Every JAX array parameter has shape documented
- [ ] Every PRNG-consuming method has "Consumes" note
- [ ] Every spec string parameter shows valid examples
- [ ] Every pytree_node=False field is noted with JIT impact
- [ ] Register method has docstring (copy from class docstring + notes)
- [ ] Type hints in code match docstring Parameter types
- [ ] No "TODO" or "FIXME" in docstrings (resolve before submitting)
- [ ] Examples are copy-pasteable and tested
- [ ] Cross-references to related classes use proper `:class:` roles

---

## Special Formatting Rules

### Code blocks in docstrings:
```python
"""
...
Examples
--------
>>> result = operator(key, population, config)
>>> print(result.values.shape)
(32, 10)

For detailed usage, see :meth:`Operator.__call__`.
"""
```

### Links to other classes:
```python
"""
...
See Also
--------
:class:`BasePopulation` : Population base class
:func:`create_operator_from_spec` : Operator factory function
"""
```

### Inline code:
```python
"""
...
The ``config`` parameter must be consistent with the population's
genes structure (shape, bounds, etc.).
...
"""
```

### Notes block for JAX behavior:
```python
"""
...
Notes
-----
JAX considerations:

- JIT-compilable: Yes/No
- Pytree-compatible: Yes/No  
- PRNG consumption: N keys of shape (2,)
- Mutates input: No
"""
```

---

## Files to Update (In Order)

**Phase 1 - Foundation:**
1. `src/malthusjax/core/base.py` ← Start here
2. `src/malthusjax/core/fitness/base.py`
3. `src/malthusjax/operators/base.py`

**Phase 2 - Operators:**
4. `src/malthusjax/operators/mutation/real.py`
5. `src/malthusjax/operators/mutation/binary.py`
6. `src/malthusjax/operators/crossover/real.py`
7. `src/malthusjax/operators/crossover/binary.py`
8. `src/malthusjax/operators/selection/*.py`

**Phase 3 - High-level:**
9. `src/malthusjax/engine/genetic_fastengine.py`
10. `src/malthusjax/composer/experiment.py`

---

## Testing Your Docstrings

```bash
# Check NumPy compliance
pydocstyle --convention=numpy src/malthusjax/core/base.py

# Build docs (warnings will show broken links/missing sections)
cd docs && make clean && make html 2>&1 | grep -i "WARNING"

# Test examples
pytest --doctest-modules src/malthusjax/core/base.py -v
```
