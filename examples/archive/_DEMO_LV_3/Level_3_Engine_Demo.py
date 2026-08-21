#!/usr/bin/env python
# coding: utf-8

# # MalthusJAX Level 3: GeneticEngine Demo
#
# This notebook demonstrates the **Level 3: Evolution Engines** of MalthusJAX.
#
# ## What is Level 3?
#
# Level 3 orchestrates complete evolutionary algorithms by combining:
# - **Level 1**: Genomes (data structures) and fitness evaluators
# - **Level 2**: Genetic operators (selection, crossover, mutation)
# - **GeneticEngine**: The orchestrator that manages evolution workflow
#
# ## Key Features
#
# 1. **Init-Phase Compilation**: Resource allocation happens once at initialization
# 2. **Entropy Management**: RNG keys allocated from pre-computed budget
# 3. **Ask/Tell Interface**: Async evolution for external evaluation
# 4. **Mutation Scheduling**: Adaptive parameter control during evolution
# 5. **JIT Compilation**: Full JAX integration for GPU acceleration

# ## Setup: Import Libraries

# In[1]:


import importlib
import time

import jax
import jax.numpy as jnp
import jax.random as jar
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

import malthusjax
from malthusjax.core.fitness.bbob_evaluator import BBOBConfig, BBOBEvaluator
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.engine import GeneticEngine, GeneticEngineParams
from malthusjax.engine.schedules import ScheduleType
from malthusjax.operators.base_ablation import (
    ablation_single_key_crossover,
    ablation_single_key_mutation,
)
from malthusjax.operators.crossover import SimulatedBinaryCrossover, UniformCrossover
from malthusjax.operators.crossover.evosax_crossover import EvosaxUniformCrossoverWrapper
from malthusjax.operators.crossover.real import UniformCrossover_injection
from malthusjax.operators.mutation import GaussianMutation
from malthusjax.operators.mutation.evosax_mutation import EvosaxGaussianWrapper
from malthusjax.operators.mutation.real import GaussianMutation_injection
from malthusjax.operators.selection import ElitePoolSelection, TournamentSelection

DIM_GEN = 500
# Plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ## Part 1: Basic Engine Initialization
#
# Let's start by creating a simple optimization problem and initializing the engine.

# In[2]:


# Set random seed for reproducibility
key = jar.PRNGKey(42)

# Define the optimization problem: minimize Sphere function
# f(x) = sum(x^2), minimum at x = [0, 0, ..., 0]
genome_config = RealGenomeConfig(
    shape=(DIM_GEN,),  # 50-dimensional optimization problem
    bounds=(-5.0, 5.0),  # Search space
)

config = BBOBConfig(maximize=False, fn_name="sphere", num_dims=DIM_GEN, seed=42)


# Create evaluator
evaluator = BBOBEvaluator.create(config)

print("✓ Genome config created")
print(f"  Shape: {genome_config.shape}")
print(f"  Bounds: {genome_config.bounds}")
print("\n✓ Evaluator created")
print(f"  Type: {type(evaluator).__name__}")


# In[3]:


# Create genetic operators
selection = ElitePoolSelection(num_selections=20, elite_k=2)
crossover = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
mutation = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)

print("✓ Genetic operators created")
print(f"  Selection: {type(selection).__name__}")
print(f"    - Selections: {selection.num_selections}")
print(f"    - Elite kept: {selection.elite_k}")
print(f"\n  Crossover: {type(crossover).__name__}")
print(f"    - Offspring per pair: {crossover.num_offspring}")
print(f"\n  Mutation: {type(mutation).__name__}")
print(f"    - Offspring per individual: {mutation.num_offspring}")
print(f"    - Mutation rate: {mutation.mutation_rate}")


# In[4]:


# Configure engine parameters
engine_params = GeneticEngineParams(pop_size=32, elitism=2, num_generations=50)

# Create engine
engine = GeneticEngine(
    engine_params=engine_params,
    genome_config=genome_config,
    evaluator=evaluator,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    enable_progress_bar=True,
)

print("✓ GeneticEngine created")
print(f"  Population size: {engine_params.pop_size}")
print(f"  Elite size: {engine_params.elitism}")
print(f"  Max generations: {engine_params.num_generations}")


# In[5]:


# Initialize engine state
state = engine.init_state(rng_key=key)

print("✓ Engine state initialized")
print(f"\n  Initial population size: {state.population.fitness.shape[0]}")
print(f"  Initial best fitness: {state.best_fitness:.6f}")
print(f"  Initial generation: {state.generation}")
print(f"\n  Best genome shape: {state.best_genome.shape}")
print(f"  Best genome: {state.best_genome}")


# ## Part 2: Evolution Loop
#
# Now let's run the evolution loop and track fitness improvements.

# In[6]:


# Run evolution loop
state = engine.init_state(key)

# Track statistics
best_fitness_history = [float(state.best_fitness)]
mean_fitness_history = [float(jnp.mean(state.population.fitness))]
std_fitness_history = [float(jnp.std(state.population.fitness))]
generation_history = [int(state.generation)]

# Run 30 generations
num_steps = 300
for i in range(num_steps):
    state, _ = engine.step(state)

    best_fitness_history.append(float(state.best_fitness))
    mean_fitness_history.append(float(jnp.mean(state.population.fitness)))
    std_fitness_history.append(float(jnp.std(state.population.fitness)))
    generation_history.append(int(state.generation))

    if (i + 1) % 100 == 0:
        print(
            f"Generation {state.generation}: Best = {state.best_fitness:.6f}, "
            f"Mean = {mean_fitness_history[-1]:.6f}"
        )

print("\n✓ Evolution complete!")
print(f"  Final best fitness: {state.best_fitness:.6f}")
print(f"  Improvement: {best_fitness_history[0] - state.best_fitness:.6f}")


# In[7]:


# Visualize evolution
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

# Plot 1: Best fitness over time
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(generation_history, best_fitness_history, "b-", linewidth=2, label="Best fitness")
ax1.fill_between(
    generation_history,
    [m - s for m, s in zip(mean_fitness_history, std_fitness_history)],
    [m + s for m, s in zip(mean_fitness_history, std_fitness_history)],
    alpha=0.2,
    label="Mean ± Std",
)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness")
ax1.set_title("Fitness Evolution Over Generations")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Mean fitness over time
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(generation_history, mean_fitness_history, "g-", linewidth=2)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Mean Fitness")
ax2.set_title("Population Mean Fitness")
ax2.grid(True, alpha=0.3)

# Plot 3: Fitness improvement rate
ax3 = fig.add_subplot(gs[1, 1])
improvement_rate = [
    best_fitness_history[i] - best_fitness_history[i + 1]
    for i in range(len(best_fitness_history) - 1)
]
ax3.bar(generation_history[:-1], improvement_rate, color="purple", alpha=0.7)
ax3.set_xlabel("Generation")
ax3.set_ylabel("Fitness Improvement")
ax3.set_title("Generation-to-Generation Improvement")
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Final population distribution (first dimension)
ax4 = fig.add_subplot(gs[2, 0])
final_pop = state.population.genes
ax4.hist(final_pop[:, 0], bins=15, color="orange", alpha=0.7, edgecolor="black")
ax4.axvline(state.best_genome[0], color="red", linestyle="--", linewidth=2, label="Best")
ax4.set_xlabel("Value (Dimension 0)")
ax4.set_ylabel("Count")
ax4.set_title("Final Population Distribution (Dim 0)")
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

# Plot 5: Best genome convergence
ax5 = fig.add_subplot(gs[2, 1])
best_genome_str = f"[{', '.join(f'{x:.3f}' for x in state.best_genome)}]"
ax5.text(0.5, 0.7, "Best Solution Found:", ha="center", fontsize=12, fontweight="bold")
ax5.text(0.5, 0.55, best_genome_str, ha="center", fontsize=10, family="monospace")
ax5.text(0.5, 0.35, f"Fitness: {state.best_fitness:.6f}", ha="center", fontsize=11)
ax5.text(
    0.5, 0.15, "Target: [0, 0, 0, 0, 0] (Fitness: 0.0)", ha="center", fontsize=10, style="italic"
)
ax5.axis("off")

plt.suptitle("GeneticEngine: Sphere Function Optimization (5D)", fontsize=14, fontweight="bold")
plt.show()

print("\n✓ Visualization complete")


# ## Part 3: Ask/Tell Interface
#
# The engine supports an async ask/tell interface for cases where evaluation happens externally.

# In[8]:


# Create a fresh state for ask/tell demo
state_asktel = engine.init_state(jar.PRNGKey(123))

print("Ask/Tell Interface Demo")
print("=" * 50)
print("\nInitial state:")
print(f"  Generation: {state_asktel.generation}")
print(f"  Best fitness: {state_asktel.best_fitness:.6f}")

# Simulate 3 ask/tell cycles
for cycle in range(3):
    print(f"\n--- Cycle {cycle + 1} ---")

    # ASK: Get next population to evaluate
    engine_with_entropy, population = engine.ask(state_asktel)
    print(f"Ask: Received population of size {population.fitness.shape[0]}")
    print(f"  Population mean fitness: {jnp.mean(population.fitness):.6f}")

    # EXTERNAL EVALUATION (simulated)
    # In a real scenario, this could be done elsewhere
    evaluated_pop = evaluator.evaluate_population(population)
    print("Evaluated: Population evaluated")
    print(f"  Evaluated mean fitness: {jnp.mean(evaluated_pop.fitness):.6f}")

    # TELL: Update state with evaluated population
    state_asktel = engine_with_entropy.tell(state_asktel, evaluated_pop)
    print("Tell: State updated")
    print(f"  New generation: {state_asktel.generation}")
    print(f"  New best fitness: {state_asktel.best_fitness:.6f}")

print("\n✓ Ask/Tell interface working correctly!")


# ## Part 4: Comparing Different Problem Types
#
# Let's optimize different benchmark functions to see how the engine adapts.

# In[9]:


# Compare different optimization problems
problems = [
    ("Sphere", BBOBConfig(maximize=False, fn_name="sphere", num_dims=DIM_GEN, seed=42)),
    ("Rosenbrock", BBOBConfig(maximize=False, fn_name="rosenbrock", num_dims=DIM_GEN, seed=42)),
]

results = {}

for problem_name, evaluator_config in problems:
    print(f"\nOptimizing {problem_name} function...")
    evaluator_fn = BBOBEvaluator.create(evaluator_config)
    engine_prob = GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator_fn,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        enable_progress_bar=False,
    )

    state_prob = engine_prob.init_state(jar.PRNGKey(42))
    best_history = [float(state_prob.best_fitness)]

    for _ in range(30):
        state_prob, _ = engine_prob.step(state_prob)
        best_history.append(float(state_prob.best_fitness))

    results[problem_name] = {
        "history": best_history,
        "final_best": state_prob.best_fitness,
        "final_genome": state_prob.best_genome,
    }

    print(f"  Final best fitness: {state_prob.best_fitness:.6f}")
    print(f"  Improvement: {best_history[0] - state_prob.best_fitness:.6f}")


# In[10]:


# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Fitness convergence comparison
colors = ["blue", "green", "red", "purple"]
for (name, data), color in zip(results.items(), colors):
    axes[0].plot(data["history"], label=name, linewidth=2, color=color)

axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Best Fitness")
axes[0].set_title("Convergence on Different Benchmark Functions")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

# Plot 2: Final improvements
names = list(results.keys())
improvements = [results[name]["history"][0] - results[name]["final_best"] for name in names]
bars = axes[1].bar(names, improvements, color=colors[: len(names)], alpha=0.7, edgecolor="black")
axes[1].set_ylabel("Total Improvement")
axes[1].set_title("Total Fitness Improvement by Problem")
axes[1].grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, improvement in zip(bars, improvements):
    height = bar.get_height()
    axes[1].text(
        bar.get_x() + bar.get_width() / 2.0, height, f"{improvement:.2f}", ha="center", va="bottom"
    )

plt.tight_layout()
plt.show()

print("✓ Comparison complete")


# ## Part 5: Mutation Strength Scheduling
#
# The engine can adapt mutation strength during evolution for better performance.

# In[11]:


# Test different mutation strength schedules
# In MalthusJAX, schedules are configured declaratively using Enums to preserve XLA performance.
schedules = {
    "Linear Decay": {
        "schedule_type": ScheduleType.LINEAR_DECAY,
        "initial_strength": 0.5,
        "final_strength": 0.1,
    },
    "Exponential Decay": {
        "schedule_type": ScheduleType.EXPONENTIAL_DECAY,
        "initial_strength": 0.5,
        "final_strength": 0.1,
    },
}

schedule_results = {}

for schedule_name, sched_kwargs in schedules.items():
    print(f"\nTesting {schedule_name} schedule...")

    engine_sched = GeneticEngine(
        engine_params=engine_params.replace(**sched_kwargs),
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection,
        crossover=crossover,
        mutation=mutation,
        enable_progress_bar=False,
    )

    state_sched = engine_sched.init_state(jar.PRNGKey(42))
    best_history = [float(state_sched.best_fitness)]

    for gen in range(30):
        state_sched, _ = engine_sched.step(state_sched)
        best_history.append(float(state_sched.best_fitness))

    schedule_results[schedule_name] = best_history
    print(f"  Final best fitness: {state_sched.best_fitness:.6f}")


# In[13]:


# Visualize schedule comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Convergence comparison
colors_sched = ["blue", "green", "red"]
for (name, history), color in zip(schedule_results.items(), colors_sched):
    axes[0].plot(history, label=name, linewidth=2, color=color, marker="o", markersize=3)

axes[0].set_xlabel("Generation")
axes[0].set_ylabel("Best Fitness")
axes[0].set_title("Convergence with Different Mutation Schedules")
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# Define helper functions to calculate expected schedule values for plotting
def linear_schedule(step, max_steps, initial=0.5, final=0.1):
    return initial - (initial - final) * (step / max_steps)


def exponential_schedule(step, max_steps, initial=0.5, final=0.1):
    return initial * ((final / initial) ** (step / max_steps))


# Plot 2: Mutation strength over time
gens = range(30)
linear_vals = [linear_schedule(g, 30) for g in gens]
exp_vals = [exponential_schedule(g, 30) for g in gens]

axes[1].plot(gens, linear_vals, "b-", linewidth=2, label="Linear", marker="o", markersize=3)
axes[1].plot(gens, exp_vals, "g-", linewidth=2, label="Exponential", marker="s", markersize=3)
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Mutation Strength")
axes[1].set_title("Mutation Strength Schedule Over Time")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Schedule comparison complete")


# ## Part 6: Understanding the Engine Internals
#
# Let's explore what happens under the hood during a single evolution step.

# In[14]:


# Create a simple state to inspect
key_inspect = jar.PRNGKey(999)
state_inspect = engine.init_state(key_inspect)

print("Engine State Structure")
print("=" * 50)
print("\nPopulation:")
print(f"  Shape: {state_inspect.population.genes.shape}")
print(f"  Fitness shape: {state_inspect.population.fitness.shape}")
print(f"  Fitness min: {jnp.min(state_inspect.population.fitness):.6f}")
print(f"  Fitness max: {jnp.max(state_inspect.population.fitness):.6f}")
print(f"  Fitness mean: {jnp.mean(state_inspect.population.fitness):.6f}")

print("\nBest Individual:")
print(f"  Genome: {state_inspect.best_genome}")
print(f"  Fitness: {state_inspect.best_fitness:.6f}")

print("\nEvolution Tracking:")
print(f"  Generation: {state_inspect.generation}")
print(f"  RNG key shape: {state_inspect.rng_key.shape}")

print("\nResource Mapping (Init-Phase Compilation):")
print(f"  Selection resource state: {type(state_inspect.resource_map.selection).__name__}")
print(f"  Crossover resource state: {type(state_inspect.resource_map.crossover).__name__}")
print(f"  Mutation resource state: {type(state_inspect.resource_map.mutation).__name__}")


# In[16]:


# Perform one step and compare
state_after = engine.step(state_inspect)[0]

print("\nState After One Evolution Step")
print("=" * 50)
print("\nPopulation:")
print(f"  Shape: {state_after.population.genes.shape}")
print(f"  Fitness min: {jnp.min(state_after.population.fitness):.6f}")
print(f"  Fitness max: {jnp.max(state_after.population.fitness):.6f}")
print(f"  Fitness mean: {jnp.mean(state_after.population.fitness):.6f}")

print("\nBest Individual:")
print(f"  Genome: {state_after.best_genome}")
print(f"  Fitness: {state_after.best_fitness:.6f}")

print("\nEvolution Tracking:")
print(f"  Generation: {state_after.generation}")

print("\nChanges:")
improvement = state_inspect.best_fitness - state_after.best_fitness
print(f"  Best fitness improvement: {improvement:.6f}")
print(f"  Generation incremented: {state_after.generation - state_inspect.generation}")
print(f"  RNG key updated: {not jnp.array_equal(state_inspect.rng_key, state_after.rng_key)}")


# ## Part 7: Performance Characteristics
#
# Let's benchmark the engine to understand its performance characteristics.

# In[17]:

# Test with different population sizes
pop_sizes = [16, 32, 64, 128]
timings = []

print("Performance Benchmark: Population Size Scaling")
print("=" * 50)

for pop_size in pop_sizes:
    params_test = engine_params.replace(pop_size=pop_size)
    engine_test = GeneticEngine(
        engine_params=params_test,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=selection.replace(num_selections=pop_size),
        crossover=crossover,
        mutation=mutation,
        enable_progress_bar=False,
    )

    state_test = engine_test.init_state(jar.PRNGKey(42))

    # Time 10 steps
    start = time.time()
    for _ in range(10):
        state_test, _ = engine_test.step(state_test)
    elapsed = time.time() - start

    avg_time = elapsed / 10
    timings.append(avg_time * 1000)  # Convert to ms

    print(f"  Pop size {pop_size:3d}: {avg_time * 1000:6.2f} ms/step")

print("\n✓ Benchmark complete")


# In[18]:


# Visualize performance scaling
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(pop_sizes, timings, "o-", linewidth=2, markersize=8, color="purple")
ax.fill_between(pop_sizes, timings, alpha=0.2, color="purple")

device_kind = jax.default_backend()


ax.set_xlabel("Population Size", fontsize=12)
ax.set_ylabel("Time per Step (ms)", fontsize=12)
ax.set_title(
    f"Engine Performance: Time Scaling with Population Size on {device_kind}",
    fontsize=13,
    fontweight="bold",
)
ax.grid(True, alpha=0.3)

# Add annotations
for pop, time_ms in zip(pop_sizes, timings):
    ax.annotate(
        f"{time_ms:.1f}ms",
        xy=(pop, time_ms),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

print("✓ Performance visualization complete")


# ## Summary
#
# You've now learned about **Level 3: Evolution Engines** in MalthusJAX!
#
# ### Key Takeaways:
#
# 1. **GeneticEngine** orchestrates complete evolutionary algorithms
# 2. **Init-Phase Compilation** allocates resources once at initialization for efficiency
# 3. **Ask/Tell Interface** supports async external evaluation workflows
# 4. **Mutation Scheduling** enables adaptive parameter control
# 5. **JAX Integration** provides JIT compilation and GPU acceleration
#
# ### Next Steps:
#
# - Explore different genome types (Binary, Categorical, Linear)
# - Try different fitness evaluators (multi-objective, constrained)
# - Implement custom selection/crossover/mutation operators
# - Use the ask/tell interface for parallel evaluation
# - Profile with JAX named_call tracing for HLO analysis
#
# ### Architecture Reminder:
#
# ```
# Level 1: Genomes & Evaluators (data structures)
#    ↑
# Level 2: Genetic Operators (selection, crossover, mutation)
#    ↑
# Level 3: GeneticEngine (orchestration & evolution loop)
# ```
#
# See `examples/` for more advanced notebooks!

# In[19]:


# === Compare two operator configurations across multiple seeds ===


def run_engine_with_seed(engine, seed, num_steps=200):
    state = engine.init_state(jar.PRNGKey(seed))
    best_history = [float(state.best_fitness)]
    for _ in range(num_steps):
        state, _ = engine.step(state)
        best_history.append(float(state.best_fitness))
    return best_history, state


def evaluate_operator_config(name, make_engine_fn, seeds=(1, 2, 3), num_steps=200):
    results = []
    histories = []
    print(f"\n=== Running: {name} (seeds={seeds}, steps={num_steps}) ===")
    for s in seeds:
        eng = make_engine_fn()
        hist, final_state = run_engine_with_seed(eng, s, num_steps=num_steps)
        results.append(float(final_state.best_fitness))
        histories.append(hist)
        print(f"{name} - seed {s}: final best = {results[-1]:.6f}")
    return {"name": name, "results": results, "histories": histories}


# Factory functions for two operator sets
def make_engine_A():
    # A: ElitePool + SimulatedBinary (default tuned)
    sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
    cross = SimulatedBinaryCrossover(num_offspring=2, eta=15.0)
    mut = GaussianMutation(num_offspring=1, mutation_rate=0.10, mutation_strength=0.5)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


def make_engine_B():
    # B: Tournament + Uniform crossover + stronger mutation for exploration
    sel = TournamentSelection(num_selections=engine_params.pop_size, tournament_size=3)
    cross = UniformCrossover(num_offspring=2, crossover_rate=0.5)
    mut = GaussianMutation(num_offspring=1, mutation_rate=0.20, mutation_strength=0.8)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


# Run comparison
SEEDS = (1, 2, 3)
STEPS = 200
resA = evaluate_operator_config("Elite+SBX", make_engine_A, seeds=SEEDS, num_steps=STEPS)
resB = evaluate_operator_config("Tournament+Uniform", make_engine_B, seeds=SEEDS, num_steps=STEPS)


# In[20]:


# === Compare two operator configurations across multiple seeds ===


def evaluate_operator_config(name, make_engine_fn, seeds=(1, 2, 3), num_steps=200):
    results = []
    histories = []
    print(f"\n=== Running: {name} (seeds={seeds}, steps={num_steps}) ===")
    for s in seeds:
        eng = make_engine_fn()
        hist, final_state = run_engine_with_seed(eng, s, num_steps=num_steps)
        results.append(float(final_state.best_fitness))
        histories.append(hist)
        print(f"{name} - seed {s}: final best = {results[-1]:.6f}")
    return {"name": name, "results": results, "histories": histories}


# Factory functions for two operator sets
def make_engine_A():
    # A: ElitePool + SimulatedBinary (default tuned)
    sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
    cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)
    mut = GaussianMutation(num_offspring=1, mutation_strength=0.8, mutation_rate=1.0)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


def make_engine_B():
    # A: ElitePool + SimulatedBinary (default tuned)
    sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
    cross = UniformCrossover_injection(num_offspring=1, crossover_rate=0.5)
    mut = GaussianMutation_injection(num_offspring=1, mutation_strength=0.8, mutation_rate=1.0)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


# Run comparison
SEEDS = [i for i in range(1, 30)]
STEPS = 200
resA = evaluate_operator_config("mjx ", make_engine_A, seeds=SEEDS, num_steps=STEPS)
resB = evaluate_operator_config("evosax", make_engine_B, seeds=SEEDS, num_steps=STEPS)


# In[21]:


# Boxplot-only comparison of final best fitness across operator configs
# If `resA`/`resB` are not defined, run quick experiments to produce them.

try:
    finalsA = np.array(resA["results"])
    finalsB = np.array(resB["results"])
    names = [resA["name"], resB["name"]]
except Exception:
    print("resA/resB not found - running quick experiments (seeds=(1,2,3), steps=200)...")
    resA = evaluate_operator_config(
        "mjx ", make_engine_A, seeds=[i for i in range(1, 30)], num_steps=200
    )
    resB = evaluate_operator_config(
        "evosax", make_engine_B, seeds=[i for i in range(1, 30)], num_steps=200
    )
    finalsA = np.array(resA["results"])
    finalsB = np.array(resB["results"])
    names = [resA["name"], resB["name"]]

finals = [finalsA, finalsB]
colors = ["C0", "C1"]

fig, ax = plt.subplots(figsize=(7, 5))
box = ax.boxplot(finals, tick_labels=names, patch_artist=True, medianprops=dict(color="black"))

# Color boxes individually
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay individual sample points for clarity (jitter on x-axis)
for i, arr in enumerate(finals, start=1):
    xs = np.random.normal(loc=i, scale=0.04, size=len(arr))
    ax.scatter(xs, arr, color="k", zorder=10, alpha=0.9)

# Annotate mean and std above each box
for i, arr in enumerate(finals, start=1):
    m, s = arr.mean(), arr.std()
    y = arr.max() + (arr.max() - arr.min()) * 0.08
    ax.text(i, y, f"μ={m:.3g}\nσ={s:.3g}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Final best fitness (lower is better)")
ax.set_title("Operator Comparison — Final Best Fitness (Boxplot across seeds)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ## Part 8: Ablation Study - Key Budgeting vs. On-Demand Splitting
#
# Let's test whether MalthusJAX's sophisticated ResourceMapper key budgeting
# system provides performance benefits compared to simple on-demand key splitting.

# In[22]:


# Import ablation decorators
importlib.reload(malthusjax.operators.base_ablation)

print("Key Budgeting Ablation Study")
print("=" * 50)
print("Testing whether ResourceMapper's sophisticated key budgeting")
print("provides performance benefits vs. simple on-demand key splitting")
print()


# Create ablation versions of operators using decorators
@ablation_single_key_mutation
class GaussianMutation_ablation(GaussianMutation):
    """Ablation version: bypasses ResourceMapper budgeting."""

    pass


@ablation_single_key_crossover
class UniformCrossover_ablation(UniformCrossover):
    """Ablation version: bypasses ResourceMapper budgeting."""

    pass


print("✓ Ablation operators created using decorators")
print("  Original operators: Use ResourceMapper pre-allocated keys")
print("  Ablation operators: Use single-key on-demand splitting")


# In[23]:


# Demonstrate key budgeting difference
print("\nKey Budget Comparison:")
print("=" * 30)

# Original operators with ResourceMapper budgeting
orig_mut = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
orig_cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)

# Ablation operators with single-key approach
abl_mut = GaussianMutation_ablation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
abl_cross = UniformCrossover_ablation(num_offspring=1, crossover_rate=0.5)

# Set population size for budgeting
pop_size = 32
orig_mut = orig_mut.set_input_length(pop_size)
orig_cross = orig_cross.set_input_length(pop_size)
abl_mut = abl_mut.set_input_length(pop_size)
abl_cross = abl_cross.set_input_length(pop_size)

# Check key budgets
input_shape = (pop_size,)
orig_mut_keys = orig_mut.num_keys(input_shape)
orig_cross_keys = orig_cross.num_keys(input_shape)
abl_mut_keys = abl_mut.num_keys(input_shape)
abl_cross_keys = abl_cross.num_keys(input_shape)

print("Mutation keys needed:")
print(f"  Original (ResourceMapper): {orig_mut_keys}")
print(f"  Ablation (Single-key):     {abl_mut_keys}")
print(f"  Reduction factor:          {orig_mut_keys / abl_mut_keys:.1f}x")

print("\nCrossover keys needed:")
print(f"  Original (ResourceMapper): {orig_cross_keys}")
print(f"  Ablation (Single-key):     {abl_cross_keys}")
print(f"  Reduction factor:          {orig_cross_keys / abl_cross_keys:.1f}x")

total_orig = orig_mut_keys + orig_cross_keys
total_abl = abl_mut_keys + abl_cross_keys
print("\nTotal keys per generation:")
print(f"  Original: {total_orig}")
print(f"  Ablation: {total_abl}")
print(f"  Memory reduction: {total_orig / total_abl:.1f}x")


# In[25]:


# Performance comparison: Original vs Ablation
print("\nPerformance Benchmark: ResourceMapper vs. Single-Key")
print("=" * 60)


def make_original_engine():
    """Engine with ResourceMapper budgeting (current architecture)."""
    sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
    cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)
    mut = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


def make_ablation_engine():
    """Engine with single-key ablation (bypass ResourceMapper)."""
    sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
    cross = UniformCrossover_ablation(num_offspring=1, crossover_rate=0.5)
    mut = GaussianMutation_ablation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


# Warm up JIT compilation
print("Warming up JIT compilation...")
orig_engine = make_original_engine()
abl_engine = make_ablation_engine()

orig_state = orig_engine.init_state(jar.PRNGKey(42))
abl_state = abl_engine.init_state(jar.PRNGKey(42))

# Run a few steps to trigger compilation
for _ in range(3):
    orig_state, _ = orig_engine.step(orig_state)
    abl_state, _ = abl_engine.step(abl_state)

print("JIT compilation complete - starting benchmarks...")


# In[26]:


# Benchmark timing performance
def benchmark_engine(engine_factory, name, num_steps=100, num_trials=5):
    """Benchmark engine performance over multiple trials."""
    times = []

    for trial in range(num_trials):
        engine = engine_factory()
        state = engine.init_state(jar.PRNGKey(42 + trial))

        start_time = time.perf_counter()
        for _ in range(num_steps):
            state, _ = engine.step(state)
        end_time = time.perf_counter()

        elapsed = end_time - start_time
        times.append(elapsed)
        print(
            f"  {name} Trial {trial + 1}: {elapsed:.4f}s ({elapsed * 1000 / num_steps:.2f} ms/step)"
        )

    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, times


# Run performance benchmarks
num_benchmark_steps = 100
num_trials = 3

print(f"\nBenchmarking {num_benchmark_steps} steps × {num_trials} trials:")
print("-" * 50)

print("Original ResourceMapper:")
orig_avg, orig_std, orig_times = benchmark_engine(
    make_original_engine, "Original", num_benchmark_steps, num_trials
)

print("\nAblation Single-Key:")
abl_avg, abl_std, abl_times = benchmark_engine(
    make_ablation_engine, "Ablation", num_benchmark_steps, num_trials
)

# Calculate performance difference
speedup = orig_avg / abl_avg if abl_avg < orig_avg else -(abl_avg / orig_avg)
ms_per_step_orig = orig_avg * 1000 / num_benchmark_steps
ms_per_step_abl = abl_avg * 1000 / num_benchmark_steps

print("\nPerformance Summary:")
print("=" * 30)
print(
    "Original (ResourceMapper):  "
    f"{ms_per_step_orig:.3f} ± {orig_std * 1000 / num_benchmark_steps:.3f} ms/step"
)
print(
    "Ablation (Single-Key):      "
    f"{ms_per_step_abl:.3f} ± {abl_std * 1000 / num_benchmark_steps:.3f} ms/step"
)
if speedup > 0:
    print(f"Performance improvement:    {speedup:.2f}x faster")
else:
    print(f"Performance degradation:    {-speedup:.2f}x slower")


# In[27]:


# Functional equivalence test
print("\nFunctional Equivalence Test:")
print("=" * 35)
print("Verifying both approaches produce equivalent optimization results...")


def run_equivalence_test(num_seeds=5, num_steps=50):
    """Test that both approaches produce similar optimization results."""
    orig_results = []
    abl_results = []

    for seed in range(num_seeds):
        # Test original engine
        orig_eng = make_original_engine()
        orig_state = orig_eng.init_state(jar.PRNGKey(seed))
        for _ in range(num_steps):
            orig_state, _ = orig_eng.step(orig_state)
        orig_results.append(float(orig_state.best_fitness))

        # Test ablation engine
        abl_eng = make_ablation_engine()
        abl_state = abl_eng.init_state(jar.PRNGKey(seed))
        for _ in range(num_steps):
            abl_state, _ = abl_eng.step(abl_state)
        abl_results.append(float(abl_state.best_fitness))

    return orig_results, abl_results


# Run equivalence test
orig_fitness, abl_fitness = run_equivalence_test(num_seeds=5, num_steps=50)

orig_mean = np.mean(orig_fitness)
abl_mean = np.mean(abl_fitness)
orig_std = np.std(orig_fitness)
abl_std = np.std(abl_fitness)

print("\nOptimization Quality (lower is better):")
print(f"  Original: {orig_mean:.6f} ± {orig_std:.6f}")
print(f"  Ablation: {abl_mean:.6f} ± {abl_std:.6f}")
print(f"  Difference: {abs(orig_mean - abl_mean):.6f}")

quality_diff_percent = abs(orig_mean - abl_mean) / abs(orig_mean) * 100
print(f"  Quality difference: {quality_diff_percent:.3f}%")

if quality_diff_percent < 1.0:
    print("  ✓ Functional equivalence confirmed (< 1% difference)")
else:
    print("  ⚠ Significant quality difference detected")


# In[ ]:


# Visualize ablation study results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Ablation Study: ResourceMapper vs. Single-Key Splitting", fontsize=14, fontweight="bold"
)

# Plot 1: Performance comparison
ax1 = axes[0, 0]
methods = ["ResourceMapper\n(Original)", "Single-Key\n(Ablation)"]
times_ms = [ms_per_step_orig, ms_per_step_abl]
stds_ms = [orig_std * 1000 / num_benchmark_steps, abl_std * 1000 / num_benchmark_steps]
colors = ["#2E86AB", "#A23B72"]

bars = ax1.bar(
    methods, times_ms, yerr=stds_ms, capsize=5, color=colors, alpha=0.7, edgecolor="black"
)
ax1.set_ylabel("Time per Step (ms)")
ax1.set_title("Performance Comparison")
ax1.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, time_val in zip(bars, times_ms):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + max(stds_ms) * 0.1,
        f"{time_val:.2f}ms",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Plot 2: Key budget comparison
ax2 = axes[0, 1]
budget_orig = [orig_mut_keys, orig_cross_keys]
budget_abl = [abl_mut_keys, abl_cross_keys]
x = np.arange(2)
width = 0.35

bars1 = ax2.bar(
    x - width / 2, budget_orig, width, label="ResourceMapper", color=colors[0], alpha=0.7
)
bars2 = ax2.bar(x + width / 2, budget_abl, width, label="Single-Key", color=colors[1], alpha=0.7)

ax2.set_ylabel("Keys Required")
ax2.set_title("Memory Usage: Key Budget per Generation")
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(["Mutation", "Crossover"])
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.1,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Plot 3: Optimization quality comparison
ax3 = axes[1, 0]
fitness_data = [orig_fitness, abl_fitness]
box_plot = ax3.boxplot(
    fitness_data, labels=methods, patch_artist=True, medianprops=dict(color="black", linewidth=2)
)

for patch, color in zip(box_plot["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax3.set_ylabel("Final Best Fitness")
ax3.set_title("Optimization Quality Comparison")
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Architecture comparison diagram
ax4 = axes[1, 1]
ax4.axis("off")
ax4.set_title("Architecture Comparison")

# Text comparison
comparison_text = f"""
ResourceMapper Architecture:
• Pre-allocates {total_orig} keys/generation
• Static memory allocation
• Complex budgeting logic
• Time: {ms_per_step_orig:.2f} ms/step

Single-Key Architecture:
• Uses {total_abl} key/generation
• Dynamic key splitting
• Simple on-demand allocation
• Time: {ms_per_step_abl:.2f} ms/step

Key Findings:
• Memory reduction: {total_orig / total_abl:.0f}x fewer keys
• Performance: {"Ablation faster" if abl_avg < orig_avg else "Original faster"}
• Quality: Equivalent (Δ = {quality_diff_percent:.2f}%)
"""

ax4.text(
    0.05,
    0.95,
    comparison_text,
    transform=ax4.transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5),
)

plt.tight_layout()
plt.show()

print("\n✓ Ablation study visualization complete")


# In[29]:


# Quantitative Analysis: Fitness Differences Across Genome Dimensions
print("Quantitative Fitness Analysis: ResourceMapper vs. Single-Key")
print("=" * 65)
print("Testing genome dimensions: 10, 50, 100 with 30 runs each")
print()


def run_dimension_experiment(dim, num_runs=30, num_steps=100):
    """Run experiments for a specific dimension size."""
    print(f"Running dimension {dim}D experiment ({num_runs} runs, {num_steps} steps each)...")

    # Create configs for this dimension
    dim_genome_config = RealGenomeConfig(shape=(dim,), bounds=(-5.0, 5.0))
    dim_bbob_config = BBOBConfig(maximize=False, fn_name="sphere", num_dims=dim, seed=42)
    dim_evaluator = BBOBEvaluator.create(dim_bbob_config)

    # Factory functions for this dimension
    def make_orig_engine():
        sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
        cross = UniformCrossover(num_offspring=1, crossover_rate=0.5)
        mut = GaussianMutation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
        return GeneticEngine(
            engine_params=engine_params,
            genome_config=dim_genome_config,
            evaluator=dim_evaluator,
            selection=sel,
            crossover=cross,
            mutation=mut,
            enable_progress_bar=False,
        )

    def make_abl_engine():
        sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
        cross = UniformCrossover_ablation(num_offspring=1, crossover_rate=0.5)
        mut = GaussianMutation_ablation(num_offspring=1, mutation_rate=0.1, mutation_strength=0.5)
        return GeneticEngine(
            engine_params=engine_params,
            genome_config=dim_genome_config,
            evaluator=dim_evaluator,
            selection=sel,
            crossover=cross,
            mutation=mut,
            enable_progress_bar=False,
        )

    orig_results = []
    abl_results = []

    for run in range(num_runs):
        seed = 100 + run  # Consistent seed offset

        # Test original engine
        orig_eng = make_orig_engine()
        orig_state = orig_eng.init_state(jar.PRNGKey(seed))
        for _ in range(num_steps):
            orig_state, _ = orig_eng.step(orig_state)
        orig_results.append(float(orig_state.best_fitness))

        # Test ablation engine
        abl_eng = make_abl_engine()
        abl_state = abl_eng.init_state(jar.PRNGKey(seed))
        for _ in range(num_steps):
            abl_state, _ = abl_eng.step(abl_state)
        abl_results.append(float(abl_state.best_fitness))

        # Progress indicator
        if (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{num_runs} runs")

    return orig_results, abl_results


# Run experiments for different dimensions
dimensions = [10, 50, 100, 300]
experiment_results = {}

for dim in dimensions:
    orig_fitness, abl_fitness = run_dimension_experiment(dim, num_runs=30, num_steps=100)
    experiment_results[dim] = {"original": orig_fitness, "ablation": abl_fitness}

print("\n✓ All dimension experiments completed")


# In[ ]:


# Statistical Analysis and Visualization of Dimension Results
print("\nStatistical Analysis of Fitness Differences")
print("=" * 45)

# Create summary statistics table
summary_stats = []
for dim in dimensions:
    orig_data = np.array(experiment_results[dim]["original"])
    abl_data = np.array(experiment_results[dim]["ablation"])

    # Calculate statistics
    orig_mean, orig_std = orig_data.mean(), orig_data.std()
    abl_mean, abl_std = abl_data.mean(), abl_data.std()
    diff_mean = abl_mean - orig_mean
    diff_percent = (diff_mean / abs(orig_mean)) * 100

    # Statistical significance test (paired t-test)
    t_stat, p_value = stats.ttest_rel(abl_data, orig_data)

    summary_stats.append(
        {
            "Dimension": f"{dim}D",
            "Original_Mean": orig_mean,
            "Original_Std": orig_std,
            "Ablation_Mean": abl_mean,
            "Ablation_Std": abl_std,
            "Difference": diff_mean,
            "Diff_Percent": diff_percent,
            "P_Value": p_value,
            "Significant": p_value < 0.05,
        }
    )

# Print summary table
df_stats = pd.DataFrame(summary_stats)
print("\nFitness Comparison Summary (30 runs per dimension):")
print("-" * 80)
for _, row in df_stats.iterrows():
    print(
        f"{row['Dimension']:>4} | Original: {row['Original_Mean']:8.3f} "
        f"± {row['Original_Std']:6.3f}"
    )
    print(f"     | Ablation: {row['Ablation_Mean']:8.3f} ± {row['Ablation_Std']:6.3f}")
    print(f"     | Difference: {row['Difference']:7.3f} ({row['Diff_Percent']:+5.2f}%)")
    significance = (
        "***"
        if row["P_Value"] < 0.001
        else "**"
        if row["P_Value"] < 0.01
        else "*"
        if row["P_Value"] < 0.05
        else "n.s."
    )
    print(f"     | Significance: p={row['P_Value']:.4f} {significance}")
    print("-" * 80)

# Comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(
    "Quantitative Analysis: Fitness Differences Across Genome Dimensions",
    fontsize=14,
    fontweight="bold",
)

# Plot 1: Box plots comparing distributions
ax1 = axes[0, 0]
all_orig_data = []
all_abl_data = []
dim_labels = []
positions = []

pos = 1
for i, dim in enumerate(dimensions):
    orig_data = experiment_results[dim]["original"]
    abl_data = experiment_results[dim]["ablation"]

    bp1 = ax1.boxplot(
        orig_data,
        positions=[pos],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#2E86AB", alpha=0.7),
        medianprops=dict(color="black"),
    )
    bp2 = ax1.boxplot(
        abl_data,
        positions=[pos + 1],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#A23B72", alpha=0.7),
        medianprops=dict(color="black"),
    )

    ax1.text(
        pos + 0.5,
        max(max(orig_data), max(abl_data)) * 1.1,
        f"{dim}D",
        ha="center",
        fontweight="bold",
    )
    pos += 3

ax1.set_ylabel("Final Best Fitness")
ax1.set_title("Distribution Comparison Across Dimensions")
ax1.grid(True, alpha=0.3, axis="y")
ax1.set_xticks([])

orig_patch = mpatches.Patch(color="#2E86AB", alpha=0.7, label="Original")
abl_patch = mpatches.Patch(color="#A23B72", alpha=0.7, label="Ablation")
ax1.legend(handles=[orig_patch, abl_patch], loc="upper right")

# Plot 2: Mean fitness with error bars
ax2 = axes[0, 1]
dims_str = [f"{d}D" for d in dimensions]
orig_means = [np.mean(experiment_results[d]["original"]) for d in dimensions]
orig_stds = [np.std(experiment_results[d]["original"]) for d in dimensions]
abl_means = [np.mean(experiment_results[d]["ablation"]) for d in dimensions]
abl_stds = [np.std(experiment_results[d]["ablation"]) for d in dimensions]

x_pos = np.arange(len(dimensions))
width = 0.35

bars1 = ax2.bar(
    x_pos - width / 2,
    orig_means,
    width,
    yerr=orig_stds,
    capsize=5,
    label="Original",
    color="#2E86AB",
    alpha=0.7,
    edgecolor="black",
)
bars2 = ax2.bar(
    x_pos + width / 2,
    abl_means,
    width,
    yerr=abl_stds,
    capsize=5,
    label="Ablation",
    color="#A23B72",
    alpha=0.7,
    edgecolor="black",
)

ax2.set_xlabel("Genome Dimension")
ax2.set_ylabel("Mean Final Best Fitness")
ax2.set_title("Mean Performance Comparison")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(dims_str)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: Percentage differences
ax3 = axes[1, 0]
diff_percents = [
    (
        (np.mean(experiment_results[d]["ablation"]) - np.mean(experiment_results[d]["original"]))
        / abs(np.mean(experiment_results[d]["original"]))
    )
    * 100
    for d in dimensions
]

colors = ["green" if dp < 0 else "red" for dp in diff_percents]
bars = ax3.bar(dims_str, diff_percents, color=colors, alpha=0.7, edgecolor="black")

ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
ax3.set_xlabel("Genome Dimension")
ax3.set_ylabel("Fitness Difference (%)")
ax3.set_title("Relative Performance Difference\n(Negative = Original Better)")
ax3.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, pct in zip(bars, diff_percents):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + (0.1 if height > 0 else -0.3),
        f"{pct:+.2f}%",
        ha="center",
        va="bottom" if height > 0 else "top",
        fontweight="bold",
    )

# Plot 4: Statistical significance and effect sizes
ax4 = axes[1, 1]
p_values = [
    stats.ttest_rel(experiment_results[d]["ablation"], experiment_results[d]["original"])[1]
    for d in dimensions
]
effect_sizes = [
    abs(np.mean(experiment_results[d]["ablation"]) - np.mean(experiment_results[d]["original"]))
    / np.sqrt(
        (np.var(experiment_results[d]["ablation"]) + np.var(experiment_results[d]["original"])) / 2
    )
    for d in dimensions
]

# Significance levels
colors_sig = [
    "red" if p < 0.001 else "orange" if p < 0.01 else "yellow" if p < 0.05 else "gray"
    for p in p_values
]

bars = ax4.bar(dims_str, effect_sizes, color=colors_sig, alpha=0.7, edgecolor="black")
ax4.set_xlabel("Genome Dimension")
ax4.set_ylabel("Effect Size (Cohen's d)")
ax4.set_title("Statistical Significance & Effect Size")
ax4.grid(True, alpha=0.3, axis="y")

# Add significance annotations
for i, (bar, p, es) in enumerate(zip(bars, p_values, effect_sizes)):
    sig_text = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        es + 0.05,
        f"{sig_text}\np={p:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Add legend for significance levels
sig_legend_elements = [
    mpatches.Patch(color="red", alpha=0.7, label="p < 0.001 (***)"),
    mpatches.Patch(color="orange", alpha=0.7, label="p < 0.01 (**)"),
    mpatches.Patch(color="yellow", alpha=0.7, label="p < 0.05 (*)"),
    mpatches.Patch(color="gray", alpha=0.7, label="p ≥ 0.05 (n.s.)"),
]
ax4.legend(handles=sig_legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()

print("\n✓ Quantitative analysis visualization complete")


# ### Ablation Study Insights
#
# The ablation study reveals important insights about MalthusJAX's architecture:
#
# **Memory Efficiency**: The ResourceMapper pre-allocates significantly more keys than needed,
# while the single-key approach dynamically splits only what's required.
#
# **Performance Trade-offs**:
# - If ablation is **faster**: The ResourceMapper's complexity adds overhead without benefits
# - If original is **faster**: Static allocation and batching provide XLA optimization advantages
#
# **Functional Equivalence**: Both approaches produce statistically equivalent optimization results,
# confirming that the key management strategy doesn't affect algorithm correctness.
#
# **Design Implications**: The results help validate whether the sophisticated budgeting system
# justifies its complexity or if simpler approaches would be preferable for future development.

# In[31]:


# === Run Evosax (SimpleGA) ===
# Place this cell after you've defined DIM_GEN and engine_params in the notebook.
try:
    from evosax.algorithms import SimpleGA
    from evosax.problems import BBOBProblem
except ImportError:
    raise ImportError(
        "evosax not installed in this env. Install with `pip install evosax` and re-run."
    )

# Problem & strategy setup (match Malthus experiment: sphere, same dim & pop)
pop_size = engine_params.pop_size
dim = DIM_GEN
seed = 42

evx_problem = BBOBProblem("sphere", num_dims=dim, seed=42)
rng = jar.PRNGKey(seed)
r_init, rng = jar.split(rng)

# Init population (uniformly within [-5,5]) & initial fitness
init_x = jax.random.uniform(r_init, (pop_size, dim), minval=-5.0, maxval=5.0)
init_fit = jnp.full((pop_size,), jnp.inf)

# Create SimpleGA strategy and params
evx_strategy = SimpleGA(population_size=pop_size, solution=init_x[0])  # solution template
evx_params = evx_strategy.default_params
evx_state = evx_strategy.init(r_init, init_x, init_fit, evx_params)
prob_state = evx_problem.init(r_init)

# Run ask/eval/tell loop
num_steps = 200  # adjust as desired
evx_best_history = [float(getattr(evx_state, "best_fitness", jnp.inf))]

for gen in range(num_steps):
    rng, rng_step = jar.split(rng)
    x, evx_state = evx_strategy.ask(rng_step, evx_state, evx_params)
    fitness, prob_state, _ = evx_problem.eval(rng_step, x, prob_state)
    evx_state, _ = evx_strategy.tell(rng_step, x, fitness, evx_state, evx_params)
    evx_best_history.append(float(getattr(evx_state, "best_fitness", jnp.nan)))
    if (gen + 1) % 50 == 0:
        print(f"[Evosax] Gen {gen + 1:4d}: best = {evx_best_history[-1]:.6f}")

print(f"\n✓ Evosax run complete. Final best fitness: {evx_best_history[-1]:.6f}")


# In[ ]:


# === Malthus vs Evosax: direct comparison cell ===

# Parameters (tweak as needed)
SEEDS = list(range(1, 11))  # seeds to run
STEPS = 200  # generations per run
POP_SIZE = engine_params.pop_size
DIM = DIM_GEN


# Helper: run Malthus engine factory (reuse your make_original_engine or define inline)
def run_malthus(engine_factory, seed, steps=STEPS):
    eng = engine_factory()
    state = eng.init_state(jax.random.key(seed))
    history = [-float(state.best_fitness)]
    for _ in range(steps):
        state, _ = eng.step(state)
        history.append(-float(state.best_fitness))
    return history, -float(state.best_fitness)


# Helper: run Evosax SimpleGA (uses same problem + dim)
def run_evosax(seed, pop_size=POP_SIZE, dim=DIM, steps=STEPS):
    try:
        from evosax.algorithms import SimpleGA
        from evosax.problems import BBOBProblem
    except Exception as e:
        raise ImportError(
            "evosax not available: install `pip install evosax` to run this cell"
        ) from e

    rng = jax.random.key(seed)
    r_init, rng = jar.split(rng)
    # init population and problem state
    evx_problem = BBOBProblem("sphere", num_dims=dim, seed=42)
    init_x = jax.random.uniform(r_init, (pop_size, dim), minval=-5.0, maxval=5.0)
    prob_state = evx_problem.init(r_init)
    init_fit, prob_state, _ = evx_problem.eval(r_init, init_x, prob_state)

    # SimpleGA strategy
    evx_strategy = SimpleGA(population_size=pop_size, solution=init_x[0])
    evx_params = evx_strategy.default_params
    evx_state = evx_strategy.init(r_init, init_x, init_fit, evx_params)

    history = [float(getattr(evx_state, "best_fitness", jnp.min(init_fit)))]
    for gen in range(steps):
        rng, rng_step = jar.split(rng)
        x, evx_state = evx_strategy.ask(rng_step, evx_state, evx_params)
        fitness, prob_state, _ = evx_problem.eval(rng_step, x, prob_state)
        evx_state, _ = evx_strategy.tell(rng_step, x, fitness, evx_state, evx_params)
        history.append(float(getattr(evx_state, "best_fitness", jnp.nan)))
    return history, float(history[-1])


# Choose the Malthus engine factory you'd like to compare (ResourceMapper original)
def make_malthus_engine():
    sel = ElitePoolSelection(num_selections=engine_params.pop_size, elite_k=2)
    cross = EvosaxUniformCrossoverWrapper(num_offspring=1, crossover_rate=0.5)
    mut = EvosaxGaussianWrapper(num_offspring=1, mutation_strength=0.5)
    return GeneticEngine(
        engine_params=engine_params,
        genome_config=genome_config,
        evaluator=evaluator,
        selection=sel,
        crossover=cross,
        mutation=mut,
        enable_progress_bar=False,
    )


# Run experiments
malthus_finals = []
evosax_finals = []
malthus_histories = []
evosax_histories = []

print(f"Running comparison over {len(SEEDS)} seeds × {STEPS} steps ...")
start_all = time.perf_counter()
for s in SEEDS:
    # Malthus
    mhist, mfinal = run_malthus(make_malthus_engine, s, STEPS)
    malthus_histories.append(mhist)
    malthus_finals.append(mfinal)

    # Evosax
    ehist, efinal = run_evosax(s, pop_size=POP_SIZE, dim=DIM, steps=STEPS)
    evosax_histories.append(ehist)
    evosax_finals.append(efinal)

    print(f"  seed={s:2d}: Malthus={mfinal:.6f}  |  Evosax={efinal:.6f}")
end_all = time.perf_counter()
print(f"\nAll runs finished in {end_all - start_all:.2f}s")

# Statistics
m_mean, m_std = np.mean(malthus_finals), np.std(malthus_finals)
e_mean, e_std = np.mean(evosax_finals), np.std(evosax_finals)
t_stat, p_val = stats.ttest_rel(malthus_finals, evosax_finals)

print("\nSummary (final best fitness):")
print(f"  Malthus: mean={m_mean:.6f}, std={m_std:.6f}")
print(f"  Evosax : mean={e_mean:.6f}, std={e_std:.6f}")
print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.3e}  (two-sided)")

# Plot: boxplot of final bests
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot(
    [malthus_finals, evosax_finals],
    tick_labels=["Malthus", "Evosax"],
    patch_artist=True,
)
axes[0].set_title("Final Best Fitness Distribution")
axes[0].set_ylabel("Final best fitness (lower is better)")
axes[0].grid(axis="y", alpha=0.3)

# Plot: average convergence (mean ± std across seeds)
m_arr = np.array(malthus_histories)
e_arr = np.array(evosax_histories)
gens = np.arange(len(m_arr[0]))

m_mean_hist = m_arr.mean(axis=0)
m_std_hist = m_arr.std(axis=0)
e_mean_hist = e_arr.mean(axis=0)
e_std_hist = e_arr.std(axis=0)

ax = axes[1]
ax.plot(gens, m_mean_hist, label="Malthus", color="C0")
ax.fill_between(gens, m_mean_hist - m_std_hist, m_mean_hist + m_std_hist, color="C0", alpha=0.2)
ax.plot(gens, e_mean_hist, label="Evosax", color="C1")
ax.fill_between(gens, e_mean_hist - e_std_hist, e_mean_hist + e_std_hist, color="C1", alpha=0.2)
ax.set_title("Mean Convergence (across seeds)")
ax.set_xlabel("Generation")
ax.set_ylabel("Best fitness")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:
