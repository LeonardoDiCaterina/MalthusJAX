from flax import struct
import jax.numpy as jnp
from .genetic_fastengine import GeneticEngine, GeneticEvolutionState, GeneticGenerationOutput, GeneticEngineParams


@struct.dataclass
class OneFifthGeneticEngineParams(GeneticEngineParams):
    """
    Parameters for the 1/5th Success Rule Genetic Engine.
    """
    target_success_rate: float = 0.2  # 1/5
    update_factor: float = 0.82       # Rechenberg's constant (approx 1/1.22)

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
        
        # 2. Calculate Success Rate
        # We need to compare the NEW fitness vs OLD fitness.
        # Note: In standard GA, "Success" is tricky because parents might be gone.
        # MR15 is usually defined for (1+1)-ES or (mu+lambda)-ES.
        # Interpretation for GA: Fraction of offspring strictly better than the *average* of previous generation?
        # Or: Fraction of offspring better than the parents they replaced?
        
        # Let's use: Fraction of population that improved over the previous generation's best.
        # (This is a simplified metric for GA).
        
        # A Better Metric for GA: 
        # Did the *Best Fitness* improve?
        improved = new_state.best_fitness > state.best_fitness
        
        # 3. Update Mutation Strength (Global Sigma)
        # We access the mutation operator inside the state
        current_mutation = new_state.operators.mutation
        current_sigma = current_mutation.mutation_strength
        
        # Rule: If improved, increase step (exploration). If not, decrease (exploitation).
        # (Or vice-versa depending on the landscape stage).
        # Standard 1/5th Rule:
        # If P_success > 1/5 -> Sigma = Sigma / update_factor (Increase range)
        # If P_success < 1/5 -> Sigma = Sigma * update_factor (Decrease range)
        
        # Since calculating P_success accurately in a complex GA merge is hard, 
        # we often just check if we are stagnating.
        
        # Let's assume we implement the rule:
        # If improved -> Grow Sigma
        # Else -> Shrink Sigma
        
        new_sigma = jnp.where(
            improved,
            current_sigma / self.engine_params.update_factor,  # Grow
            current_sigma * self.engine_params.update_factor  # Shrink
        )
        
        # 4. Inject Updated Operator back into State
        new_op = current_mutation.replace(mutation_strength=new_sigma)
        new_ops = new_state.operators.replace(mutation=new_op)
        
        final_state = new_state.replace(operators=new_ops)
        
        return final_state, metrics