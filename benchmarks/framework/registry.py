from typing import Callable, Dict, Any, NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex

from .adapters import MalthusAdapter, EvosaxAdapter

# --- Malthus Components ---
from malthusjax.engine.genetic_fastengine import (
    GeneticEngine, 
    GeneticEngineParams, 
    GeneticEvolutionState, 
    GeneticGenerationOutput,
    traceable
)
from malthusjax.engine.MR15_GA import OneFifthGeneticEngine, OneFifthGeneticEngineParams
from malthusjax.engine.differential_engine import DifferentialEvolutionEngine
from malthusjax.operators.mutation.real import GaussianMutation, DifferentialMutation
from malthusjax.operators.crossover.real import UniformCrossover, BinomialCrossover
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.base import BaseSelection

# --- Malthus Ablation Components ---
from malthusjax.operators.mutation.ablation_mutation import AblationGaussianMutation
from malthusjax.operators.crossover.ablation_crossover import AblationUniformCrossover
from malthusjax.operators.selection.ablation_selection import AblationElitePoolSelection
from malthusjax.core.genome.real_genome import RealGenomeConfig

# --- Evosax Components ---    
from evosax.algorithms.population_based import SimpleGA, MR15_GA, DifferentialEvolution
# Helper to access Evosax internal logic for the optimization patch
from evosax.algorithms.population_based.simple_ga import crossover as es_crossover, mutation as es_mutation


# =========================================================
# REGISTRY INFRASTRUCTURE
# =========================================================

class ComparisonSpec(NamedTuple):
    name: str
    malthus_factory: Callable
    evosax_factory: Optional[Callable] = None  # Optional for Malthus-only variants
    default_hypers: Dict[str, Any] = {}

class ComparisonRegistry:
    _registry: Dict[str, ComparisonSpec] = {}

    @classmethod
    def register(cls, spec: ComparisonSpec):
        cls._registry[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> ComparisonSpec:
        if name not in cls._registry:
            raise ValueError(f"Unknown algo: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]


# =========================================================
# SPEED COMPONENTS (DEFINED INLINE FOR SAFETY)
# =========================================================

@struct.dataclass
class TournamentSelection(BaseSelection):
    """
    Tournament Selection.
    Selects the best individual from random subsets. 
    Removes the sorting bottleneck (O(N) vs O(N log N)).
    """
    tournament_size: int = struct.field(pytree_node=False, default=3)

    def num_keys(self, input_shape: tuple) -> int:
        return 1

    def __call__(self, keys: chex.Array, fitness: chex.Array) -> chex.Array:
        pop_size = fitness.shape[0]
        
        # 1. Generate Candidates (No Global Sync)
        candidates = jax.random.randint(
            keys, 
            shape=(self.num_selections, self.tournament_size), 
            minval=0, 
            maxval=pop_size
        )
        
        # 2. Retrieve Fitness
        candidate_fitness = jnp.take(fitness, candidates, axis=0)
        
        # 3. Find Winner (Argmax locally)
        # Assuming higher fitness is better (Maximization). 
        # For minimization problems (like Sphere/Rosenbrock where lower is better),
        # Ensure your fitness function negates the value (e.g. -loss).
        winner_local_indices = jnp.argmax(candidate_fitness, axis=1)
        
        # 4. Gather Winners
        selected_indices = jnp.take_along_axis(
            candidates, 
            winner_local_indices[:, None], 
            axis=1
        ).squeeze()
        
        return selected_indices


@struct.dataclass
class GeneticSpeedEngine(GeneticEngine):
    """
    The 'Speed Demon' Engine.
    1. Skips _update_hof (Removes Global Argmax Barrier).
    2. Returns minimal state updates for maximum throughput.
    """
    @traceable("Speed_Step")
    def step(self, state: GeneticEvolutionState) -> Tuple[GeneticEvolutionState, GeneticGenerationOutput]:
        
        # 1. Allocate Keys (Static)
        k_sel, k_cross, k_mut, k_next = self._allocate_entropy(state)

        # 2. Selection
        _, parent_indices = self._selection_phase(
            k_sel, state.population, state.operators, self.engine_params
        )

        # 3. Reproduction
        mutants = self._reproduction_phase(
            k_cross, k_mut, parent_indices, state.population,
            state.operators, state.resource_map
        )

        # 4. Merge (Fast Generational Replacement)
        # Truncate mutants to match population size exactly
        pop_size = state.population.genes.values.shape[0]
        truncated_genes = mutants.genes.values[:pop_size]
        next_population = state.population.replace(genes=mutants.genes.replace(values=truncated_genes))
        
        # 5. Evaluate
        evaluated_pop = self.evaluator.evaluate_population(next_population)

        # 6. Minimal State Update (NO HOF / NO ARGMAX)
        # We preserve the old best_genome blindly to avoid the check
        next_state = state.replace(
            population=evaluated_pop,
            generation=state.generation + 1,
            rng_key=k_next
        )
        
        # 7. Compute required KPIs for output
        best_fitness = jnp.max(evaluated_pop.fitness)
        mean_fitness = jnp.mean(evaluated_pop.fitness)
        
        return next_state, GeneticGenerationOutput(
            random_key=k_next,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            generation=next_state.generation
        )


# =========================================================
# EVOSAX OPTIMIZATIONS (THE PATCH)
# =========================================================

class OptimizedSimpleGA(SimpleGA):
    """
    Patched Evosax SimpleGA that removes the `searchsorted` bottleneck.
    """
    def _ask(self, key, state, params):
        # 1. Sort population (Standard)
        idx = jnp.argsort(state.fitness)
        sorted_pop = state.population[idx]
        
        # 2. Slice Elites (Optimization: View instead of search)
        elites = sorted_pop[:self.num_elites]
        
        # 3. Uniform Sample (Optimization: Integer sampling vs Weighted choice)
        rng_cross, rng_mut, rng_p1, rng_p2 = jax.random.split(key, 4)
        parents_1 = jax.random.choice(rng_p1, elites, (self.population_size,))
        parents_2 = jax.random.choice(rng_p2, elites, (self.population_size,))
        
        rng_cross_split = jax.random.split(rng_cross, self.population_size)
        rng_mut_split = jax.random.split(rng_mut, self.population_size)

        population = jax.vmap(es_crossover, in_axes=(0, 0, 0, None))(
            rng_cross_split, parents_1, parents_2, params.crossover_rate
        )

        population = jax.vmap(es_mutation, in_axes=(0, 0, None))(
            rng_mut_split, population, state.std
        )

        return population, state


# =========================================================
# FACTORY BUILDERS
# =========================================================

# --- 1. Standard Builders ---

def _build_malthus_ga(pop_size, dims, seed, hypers, problem_evaluator):
    """Standard MalthusJAX GA."""
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    
    mutation = GaussianMutation(
        mutation_rate=hypers.get('mutation_rate', 0.1),
        mutation_strength=hypers.get('sigma', 0.1)
    )
    
    crossover = UniformCrossover(
        num_offspring=2,
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = int(pop_size * elite_ratio)
    
    selection = ElitePoolSelection(
        num_selections=pop_size,
        elite_k=elite_count
    )
    
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=1,
        elitism=elite_count
    )
    
    engine = GeneticEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params
    )
    
    return MalthusAdapter(engine)

def _build_evosax_ga(pop_size, dims, seed, hypers, problem_object):
    """Standard Evosax SimpleGA (The Baseline)."""
    rng = jax.random.PRNGKey(seed)
    init_solution = problem_object.sample(rng)
    
    strategy = SimpleGA(population_size=pop_size, solution=init_solution)
    strategy.elite_ratio = hypers.get('elite_ratio', 0.5)
    es_params = strategy.default_params.replace(
        crossover_rate=hypers.get('crossover_rate', 0.5)
    ) 
    return EvosaxAdapter(strategy, es_params, problem_object, pop_size)

def _build_evosax_ga_optimized(pop_size, dims, seed, hypers, problem_object):
    """Optimized Evosax SimpleGA (The Patched Baseline)."""
    rng = jax.random.PRNGKey(seed)
    init_solution = problem_object.sample(rng)
    
    strategy = OptimizedSimpleGA(population_size=pop_size, solution=init_solution)
    strategy.elite_ratio = hypers.get('elite_ratio', 0.5)
    es_params = strategy.default_params.replace(
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    return EvosaxAdapter(strategy, es_params, problem_object, pop_size)


# --- 2. Speed Builders (The New Stuff) ---

def _build_malthus_bf16(pop_size, dims, seed, hypers, problem_object):
    """MalthusJAX using BFloat16 Memory Layout."""
    adapter = _build_malthus_ga(pop_size, dims, seed, hypers, problem_object)
    # Patch Config
    new_config = adapter.engine.genome_config.replace(dtype=jnp.bfloat16)
    adapter.engine = adapter.engine.replace(genome_config=new_config)
    return adapter

def _build_malthus_tournament(pop_size, dims, seed, hypers, problem_object):
    """MalthusJAX using Tournament Selection."""
    adapter = _build_malthus_ga(pop_size, dims, seed, hypers, problem_object)
    # Swap Selection
    new_selection = TournamentSelection(num_selections=pop_size, tourn_size=3)
    adapter.engine = adapter.engine.replace(selection=new_selection)
    return adapter

def _build_malthus_no_hof(pop_size, dims, seed, hypers, problem_object):
    """MalthusJAX using SpeedEngine (No HOF)."""
    adapter = _build_malthus_ga(pop_size, dims, seed, hypers, problem_object)
    # Swap Engine
    speed_engine = GeneticSpeedEngine(
        evaluator=adapter.engine.evaluator,
        genome_config=adapter.engine.genome_config,
        selection=adapter.engine.selection,
        crossover=adapter.engine.crossover,
        mutation=adapter.engine.mutation,
        engine_params=adapter.engine.engine_params
    )
    return MalthusAdapter(speed_engine)

def _build_malthus_speed_demon(pop_size, dims, seed, hypers, problem_object):
    """
    THE THEORETICAL LIMIT:
    1. BFloat16
    2. Tournament Selection
    3. SpeedEngine (No HOF)
    """
    adapter = _build_malthus_ga(pop_size, dims, seed, hypers, problem_object)
    
    # 1. BF16
    bf16_config = adapter.engine.genome_config.replace(dtype=jnp.bfloat16)
    
    # 2. Tournament
    tourn_selection = TournamentSelection(num_selections=pop_size, tourn_size=3)
    
    # 3. Speed Engine
    speed_engine = GeneticSpeedEngine(
        evaluator=adapter.engine.evaluator,
        genome_config=bf16_config,
        selection=tourn_selection,
        crossover=adapter.engine.crossover,
        mutation=adapter.engine.mutation,
        engine_params=adapter.engine.engine_params
    )
    return MalthusAdapter(speed_engine)


# --- 3. Ablation Builders ---

def _build_malthus_ga_ablation(pop_size, dims, seed, hypers, problem_evaluator):
    """MalthusJAX with disabled Static Resource Allocation."""
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    mutation = AblationGaussianMutation(
        mutation_rate=hypers.get('mutation_rate', 0.1),
        mutation_strength=hypers.get('sigma', 0.1),
        seed=seed
    )
    crossover = AblationUniformCrossover(
        num_offspring=2,
        crossover_rate=hypers.get('crossover_rate', 0.5),
        seed=seed
    )
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = int(pop_size * elite_ratio)
    selection = AblationElitePoolSelection(
        num_selections=pop_size,
        elite_k=elite_count,
        seed=seed
    )
    engine_params = GeneticEngineParams(
        pop_size=pop_size,
        num_generations=1,
        elitism=elite_count
    )
    engine = GeneticEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params
    )
    return MalthusAdapter(engine)


# --- 4. Special Algorithm Builders ---

def _build_evosax_mr15_ga(pop_size, dims, seed, hypers, problem_object):
    rng = jax.random.PRNGKey(seed)
    init_solution = problem_object.sample(rng)
    strategy = MR15_GA(population_size=pop_size, solution=init_solution)
    strategy.elite_ratio = hypers.get('elite_ratio', 0.5)
    return EvosaxAdapter(strategy, strategy.default_params, problem_object, pop_size)

def _build_malthus_mr15(pop_size, dims, seed, hypers, problem_evaluator):
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    mutation = GaussianMutation(
        mutation_rate=hypers.get('mutation_rate', 0.1),
        mutation_strength=hypers.get('sigma', 1.0)
    )
    crossover = UniformCrossover(
        num_offspring=2,
        crossover_rate=hypers.get('crossover_rate', 0.5)
    )
    elite_ratio = hypers.get('elite_ratio', 0.5)
    elite_count = int(pop_size * elite_ratio)
    selection = ElitePoolSelection(
        num_selections=pop_size,
        elite_k=elite_count
    )
    engine_params = OneFifthGeneticEngineParams(
        pop_size=pop_size,
        num_generations=1,
        elitism=elite_count,
        std_min=hypers.get('std_min', 0.0),
        std_max=hypers.get('std_max', float('inf')),
        std_ratio=hypers.get('std_ratio', 0.2),
    )
    engine = OneFifthGeneticEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        engine_params=engine_params
    )
    return MalthusAdapter(engine)

def _build_malthus_de(pop_size, dims, seed, hypers, problem_evaluator):
    from malthusjax.engine.base import AbstractEngineParams
    genome_config = RealGenomeConfig(length=dims, bounds=(-5.0, 5.0))
    mutation = DifferentialMutation(f_scale=hypers.get('differential_weight', 0.8))
    crossover = BinomialCrossover(num_offspring=1, crossover_rate=hypers.get('crossover_rate', 0.9))
    engine_params = AbstractEngineParams(pop_size=pop_size, num_generations=1, elitism=0)
    engine = DifferentialEvolutionEngine(
        evaluator=problem_evaluator,
        genome_config=genome_config,
        mutation=mutation,
        crossover=crossover,
        engine_params=engine_params
    )
    return MalthusAdapter(engine)

def _build_evosax_de(pop_size, dims, seed, hypers, problem_object):
    rng = jax.random.PRNGKey(seed)
    init_solution = problem_object.sample(rng)
    strategy = DifferentialEvolution(population_size=pop_size, solution=init_solution)
    es_params = strategy.default_params.replace(
        crossover_rate=hypers.get('crossover_rate', 0.9),
        differential_weight=hypers.get('differential_weight', 0.8),
        elitism=hypers.get('elitism', True)
    )
    return EvosaxAdapter(strategy, es_params, problem_object, pop_size)


# =========================================================
# REGISTRATIONS
# =========================================================

# 1. Standard Comparison
ComparisonRegistry.register(ComparisonSpec(
    name="Standard_GA",
    malthus_factory=_build_malthus_ga,
    evosax_factory=_build_evosax_ga,
    default_hypers={'mutation_rate': 0.05, 'crossover_rate': 0.6, 'sigma': 0.1, 'elite_ratio': 0.1}
))

# 2. Ablation Comparison (Dynamic vs Patched Static)
ComparisonRegistry.register(ComparisonSpec(
    name="Standard_GA_Ablation",
    malthus_factory=_build_malthus_ga_ablation,
    evosax_factory=_build_evosax_ga_optimized,
    default_hypers={'mutation_rate': 0.05, 'crossover_rate': 0.6, 'sigma': 0.1, 'elite_ratio': 0.1}
))

# 3. MR15 Comparison
ComparisonRegistry.register(ComparisonSpec(
    name="MR15_GA",
    malthus_factory=_build_malthus_mr15,
    evosax_factory=_build_evosax_mr15_ga,
    default_hypers={
        'mutation_rate': 0.1, 'crossover_rate': 0.5, 'sigma': 1.0, 
        'elite_ratio': 0.5, 'std_min': 0.0, 'std_max': float('inf'), 'std_ratio': 0.2
    }
))

# 4. DE Comparison
ComparisonRegistry.register(ComparisonSpec(
    name="Differential_Evolution",
    malthus_factory=_build_malthus_de,
    evosax_factory=_build_evosax_de,
    default_hypers={'crossover_rate': 0.9, 'differential_weight': 0.8, 'elitism': True}
))

# 5. Speed of Light Variants (Malthus Only)
ComparisonRegistry.register(ComparisonSpec(
    name="Malthus_BF16",
    malthus_factory=_build_malthus_bf16,
    default_hypers={'mutation_rate': 0.05, 'crossover_rate': 0.6, 'sigma': 0.1, 'elite_ratio': 0.1}
))

ComparisonRegistry.register(ComparisonSpec(
    name="Malthus_Tournament",
    malthus_factory=_build_malthus_tournament,
    default_hypers={'mutation_rate': 0.05, 'crossover_rate': 0.6, 'sigma': 0.1, 'elite_ratio': 0.1}
))

ComparisonRegistry.register(ComparisonSpec(
    name="Malthus_NoHOF",
    malthus_factory=_build_malthus_no_hof,
    default_hypers={'mutation_rate': 0.05, 'crossover_rate': 0.6, 'sigma': 0.1, 'elite_ratio': 0.1}
))

ComparisonRegistry.register(ComparisonSpec(
    name="Malthus_SpeedDemon",
    malthus_factory=_build_malthus_speed_demon,
    default_hypers={'mutation_rate': 0.05, 'crossover_rate': 0.6, 'sigma': 0.1, 'elite_ratio': 0.1}
))