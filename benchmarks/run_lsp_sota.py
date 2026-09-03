import os
import sys

# Ensure lsp_project components are loaded so decorators fire
import lsp.evaluator.supervised
import lsp.operators.cartesian
import lsp.operators.cartesian_crossover
import lsp.operators.crossover
import lsp.operators.mutation

from malthusjax.composer.composer import Composer
from malthusjax.composer.engine_registry import list_available as list_engines
from malthusjax.composer.genome_catalog import get_registry as get_genomes
from malthusjax.composer._registry import get_registry as get_operators

def main():
    print("Registered Engines:", list_engines())
    
    # Load and execute the declarative TOML pipeline
    # Note: we use relative path assuming we run from root directory
    toml_path = "benchmarks/supervised_learning_sota.toml"
    print(f"Loading {toml_path}...")
    
    # We must disable shared_initial_population because CGP and TensorNEAT
    # use completely different genome structures and cannot share initialization matrices.
    results = Composer.from_toml(toml_path, shared_initial_population=False)
    
    print("\n================== SUMMARY ==================")
    print(results.summary_table())
    
    plot_path = "results/supervised_sota_convergence.png"
    print(f"\nGenerating convergence plot at {plot_path}...")
    os.makedirs("results", exist_ok=True)
    results.plot_convergence(save_path=plot_path)

if __name__ == "__main__":
    main()
