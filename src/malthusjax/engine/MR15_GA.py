from flax import struct
import jax.numpy as jnp
from .genetic_fastengine import GeneticEngine, GeneticEvolutionState, GeneticGenerationOutput, GeneticEngineParams


@struct.dataclass
class OneFifthGeneticEngineParams(GeneticEngineParams):
    """
    Parameters for the 1/5th Success Rule Genetic Engine.
    """
    target_success_rate: float = 0.2  # 1/5
    # Use doubling/halving like MR15 implementation
    std_min: float = 0.0
    std_max: float = jnp.inf
    std_ratio: float = 0.2

@struct.dataclass
class OneFifthGeneticEngine(GeneticEngine):
    """
    Implements the 1/5th Success Rule.
    Adjusts mutation strength based on success rate of offspring.
    """
    
    # We override the _merge phase to calculate success rate
    def _merge(self, elites_genes, mutant_genes, old_state):
        # 1. Standard Merge Logic (Keep this)
        next_genes = super()._merge(elites_genes, mutant_genes, old_state)
        return next_genes

    # We override step to inject the parameter update
    def step(self, state: GeneticEvolutionState):
        # 1. Run Standard Step
        # This returns the state with the NEW population
        new_state, metrics = super().step(state)
        # 2. Calculate beneficial mutation rate (fraction of individuals that improved)
        # Compare new population fitness against previous population fitness element-wise.
        prev_fitness = state.population.fitness
        new_fitness = new_state.population.fitness

        # beneficial_mutation_rate: fraction of individuals with improved fitness
        beneficial_mutation_rate = jnp.mean(new_fitness < prev_fitness)

        # 3. Update Mutation Strength (Global Sigma) using MR15 rule (double / half)
        current_mutation = new_state.operators.mutation
        current_sigma = current_mutation.mutation_strength

        increase_std = beneficial_mutation_rate > self.engine_params.std_ratio
        new_sigma = jnp.where(increase_std, 2.0 * current_sigma, 0.5 * current_sigma)

        # Clip to bounds
        new_sigma = jnp.clip(new_sigma, a_min=self.engine_params.std_min, a_max=self.engine_params.std_max)

        # 4. Inject Updated Operator back into State
        new_op = current_mutation.replace(mutation_strength=new_sigma)
        new_ops = new_state.operators.replace(mutation=new_op)

        final_state = new_state.replace(operators=new_ops)

        return final_state, metrics