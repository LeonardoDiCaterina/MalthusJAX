import unittest
import jax
import jax.numpy as jnp
import jax.random as jar
import time
from flax import struct

# --- Imports (Adjusted to your project structure) ---
from malthusjax.core.genome.real_genome import RealGenomeConfig
from malthusjax.operators.selection.elite_pool import ElitePoolSelection
from malthusjax.operators.crossover.real import SimulatedBinaryCrossover
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.core.fitness.bbob_evaluator import BBOBEvaluator, BBOBConfig
from malthusjax.engine.genetic_fastengine import GeneticEngine, GeneticEngineParams, GeneticEvolutionState

class TestLevel3Engine(unittest.TestCase):
    def setUp(self):
        """Standard Setup for all tests."""
        self.key = jar.PRNGKey(42)
        
        # 1. Configuration
        self.pop_size = 100
        self.genome_shape = (10,)
        self.bounds = (-5.0, 5.0)
        self.generations = 10
        
        self.genome_config = RealGenomeConfig(
            shape=self.genome_shape, 
            bounds=self.bounds
        )
        
        self.engine_params = GeneticEngineParams(
            pop_size=self.pop_size,
            elitism=2,
            num_generations=self.generations
        )
        
        # 2. Operators
        # FIX: Use factory method to ensure 'data=None' and internal state are set correctly
        bbob_config = BBOBConfig(
            fn_name="sphere", 
            num_dims=self.genome_shape[0], 
            maximize=False
        )
        self.evaluator = BBOBEvaluator.create(bbob_config)
        
        self.selection = ElitePoolSelection(
            num_selections=self.pop_size, # Will be overridden by ResourceMap logic
            elite_k=10
        )
        self.crossover = SimulatedBinaryCrossover(
            num_offspring=2,
            eta=15.0
        )
        self.mutation = GaussianMutation(
            num_offspring=1,
            mutation_rate=0.1,
            mutation_strength=0.5
        )
        
        # 3. Engine
        self.engine = GeneticEngine(
            engine_params=self.engine_params,
            genome_config=self.genome_config,
            evaluator=self.evaluator,
            selection=self.selection,
            crossover=self.crossover,
            mutation=self.mutation,
            enable_progress_bar=False # Keep stdout clean for tests
        )

    def test_01_init_state_baking(self):
        """Test if init_state correctly compiles the plan and bakes operators."""
        print("\n[Test] Initialization & Baking")
        state = self.engine.init_state(self.key)
        
        # Check State Integrity
        self.assertIsInstance(state, GeneticEvolutionState)
        self.assertEqual(state.generation, 0)
        self.assertEqual(state.population.fitness.shape, (self.pop_size,))
        
        # Check Resource Map Integrity
        rmap = state.resource_map
        print(f"  RNG Budget per Gen: {rmap.total_rng_budget} keys")
        
        # Verify Supply/Demand Logic (Pop 100 -> Pairs 50 -> Parents 100)
        self.assertEqual(rmap.selection.output_count, 100)
        
        # Verify Baked Operators
        ops = state.operators
        self.assertEqual(ops.selection.num_selections, 100)
        self.assertEqual(ops.crossover.input_length, 50) # 50 pairs
        self.assertEqual(ops.mutation.input_length, 100) # 100 mutants

    def test_02_step_execution(self):
        """Test a single manual step execution."""
        print("\n[Test] Single Step Execution")
        state = self.engine.init_state(self.key)
        
        # JIT the step function to verify XLA compatibility
        jit_step = jax.jit(self.engine.step)
        
        start = time.time()
        final_state, metrics = jit_step(state)
        # Block to ensure execution finished
        _ = final_state.best_fitness.block_until_ready()
        duration = time.time() - start
        
        print(f"  Step Time (compile+run): {duration:.4f}s")
        
        self.assertEqual(final_state.generation, 1)
        self.assertEqual(final_state.population.genes.values.shape, (self.pop_size,) + self.genome_shape)
        
        # Check that we actually did something (fitness changed or valid)
        self.assertFalse(jnp.isnan(final_state.best_fitness))

    def test_03_closed_loop_fusion(self):
        """Test Level 3 'Closed Loop' Compilation and Fusion."""
        print("\n[Test] Level 3 Closed Loop Fusion")
        state = self.engine.init_state(self.key)
        
        # 1. Extract Optimized HLO
        # This triggers the full XLA compiler (optimize=True)
        hlo_text = self.engine.get_hlo_text(state, optimize=True, print_analysis=False)
        
        # 2. Check for Fusion
        # We expect XLA to merge operations into "%fused_computation" blocks
        fusion_count = hlo_text.count("fusion")
        print(f"  Fused Blocks: {fusion_count}")
        self.assertTrue(fusion_count > 0, "Warning: No fusion detected. Performance may be suboptimal.")
        
        # 3. Check for the Loop
        # The Python 'for' loop should become an XLA 'while' loop
        has_loop = "while" in hlo_text
        print(f"  GPU Loop Detected: {has_loop}")
        self.assertTrue(has_loop, "Critical: Python loop was NOT compiled into XLA while loop.")

    def test_04_benchmark_throughput(self):
        """Performance Sanity Check."""
        print("\n[Test] Throughput Benchmark")
        state = self.engine.init_state(self.key)
        
        # Configure a longer run
        NUM_GENS = 500
        new_params = self.engine_params.replace(num_generations=NUM_GENS)        
        
        bench_engine = self.engine.replace(engine_params=new_params)
        # Warmup (Compile)
        print("  Compiling...", end="", flush=True)
        t0 = time.time()
        final_state, _, _ = bench_engine.run(state, compile=True)
        _ = final_state.best_fitness.block_until_ready()
        print(f" Done ({time.time()-t0:.2f}s)")
        
        # Real Run
        state = bench_engine.init_state(self.key) 

        t0 = time.time()
        final_state, _, _ = bench_engine.run(state, compile=True)
        _ = final_state.best_fitness.block_until_ready()
        duration = time.time() - t0
        
        gens_per_sec = NUM_GENS / duration
        print(f"  Speed: {gens_per_sec:,.2f} gens/sec")
        
        # Expect high performance (e.g. >1000 gens/sec on GPU, >100 on CPU)
        # We set a conservative threshold to pass on CI/CD
        self.assertTrue(gens_per_sec > 50, f"Engine too slow ({gens_per_sec:.2f}). Check for Python fallback.")

    def test_05_odd_population_size(self):
        """Test the ResourceMapper fix for odd population sizes."""
        print("\n[Test] Odd Population Size (17)")
        
        # 1. Create modified parameters (New Object)
        odd_params = self.engine_params.replace(pop_size=17)
        
        # 2. Create a NEW Engine with those params (Pattern: .replace())
        bench_engine = self.engine.replace(engine_params=odd_params)
        
        # 3. Use 'bench_engine' (not self.engine) for the rest of the test
        state = bench_engine.init_state(self.key)
        rmap = state.resource_map
        
        # Check Logic: Pop 17 -> Pairs 9 (18 parents) -> Output 17
        print(f"  Pop: 17 -> Parents Needed: {rmap.selection.output_count}")
        self.assertEqual(rmap.selection.output_count, 18)
        
        # Run
        final_state, _, _ = bench_engine.run(state)
        
        # Verify Output Shape
        self.assertEqual(final_state.population.genes.values.shape[0], 17)
        print("  Odd population handled correctly.")
        
        
    def test_06_ask_tell_equivalence(self):
        """
        Verify that the Ask-Tell interface produces identical genes to the Step interface.
        This ensures the decoupled execution mode is mathematically consistent with the fused mode.
        """
        print("\n[Test] Ask-Tell vs Step Equivalence")
        
        # 1. Initialize State
        state_0 = self.engine.init_state(self.key)
        
        # --- PATH A: Fused Step ---
        # Run one generation using the standard fused step
        # Note: step() performs eval and HOF update at the end of the generation
        state_step, _ = self.engine.step(state_0)
        
        # --- PATH B: Ask-Tell ---
        # 1. Ask: Allocate entropy for the next step
        # This returns the engine with entropy buffer populated
        engine_with_entropy, _ = self.engine.ask(state_0)
        
        # 2. Tell: Execute evolutionary logic using the buffered entropy
        # Note: tell() performs HOF update at the START (using input pop) 
        # and returns an UNEVALUATED new population.
        state_tell = engine_with_entropy.tell(state_0, state_0.population)
        
        # --- COMPARISON ---
        
        # 1. Check Genomes (The most critical check)
        # The genes produced by reproduction/mutation must be identical bit-for-bit
        genes_step = state_step.population.genes.values
        genes_tell = state_tell.population.genes.values
        
        diff = jnp.abs(genes_step - genes_tell).sum()
        print(f"  Gene Difference: {diff}")
        self.assertEqual(diff, 0.0, "Ask-Tell produced different genes than Step!")
        
        # 2. Check RNG Forwarding
        # Both methods consume 'k_next' from the ResourceMap, so the resulting
        # state.rng_key must be identical to ensure future generations stay synced.
        print(f"  RNG Step: {state_step.rng_key}")
        print(f"  RNG Tell: {state_tell.rng_key}")
        self.assertTrue(jnp.array_equal(state_step.rng_key, state_tell.rng_key), 
                        "RNG state diverged between Ask-Tell and Step.")
        
        # Note on Fitness:
        # We DO NOT compare fitness or best_fitness here.
        # - 'state_step' has Evaluated fitness (computed at end of step)
        # - 'state_tell' has Unevaluated/Stale fitness (computed at start of next tell)
        # This difference is by design.
        
        print("  Equivalence verified successfully.")

if __name__ == '__main__':
    unittest.main()