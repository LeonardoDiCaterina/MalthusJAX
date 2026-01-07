import os

base_toml = """
[experiment]
name = "GECCO_{TASK}"
output_dir = "results/gecco_final/{TASK}"

[grid]
algorithms = ["Standard_GA"]
tasks = ["{TASK}"]

# DIMENSIONS: Small, Medium, Large
dimensions = [10, 50, 100]

# POPULATION SIZES: Warp Alignment Study
# Aligned (multiples of 32): 32, 64, 128, 256, 512, 1024
# Misaligned (N+1): 33, 65, 129, 257, 513, 1025
# This tests GPU resource utilization without proliferating test count
pop_sizes = [
    32, 33,           # Boundary: 1-2 warps
    64, 65,           # Boundary: 2-3 warps
    128, 129,         # Boundary: 4-5 warps
    256, 257,         # Boundary: 8-9 warps
    512, 513,         # Boundary: 16-17 warps
    1024, 1025        # Boundary: 32-33 warps
]

# UNROLL: Keep baseline only (1 is standard, no fusion overhead)
unroll_factors = [1]

# STATISTICS
repeats = 30
seeds = [i + 42 for i in range(repeats)]
generations = 2000

[grid.hyperparams]
mutation_rate = 1.0
mutation_strength = 1.0
crossover_rate = 1.0
elite_ratio = 0.5
"""

tasks = [
    "sphere",
    "rosenbrock",
    "ellipsoidal_rotated",
    "rastrigin",
    "schaffers_f7"
]

os.makedirs("configs", exist_ok=True)

print("📦 Generating Warp-Aligned Benchmark Configurations:")
for task in tasks:
    filename = f"configs/run_{task}.toml"
    content = base_toml.replace("{TASK}", task)
    
    with open(filename, "w") as f:
        f.write(content)
    
    print(f"   - Created {filename}")

print("\n✅ Done. Configuration includes warp boundary stress testing (N=32→1025).")