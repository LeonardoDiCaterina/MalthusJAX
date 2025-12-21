import os

# The "Production" Configuration Template
base_toml = """
[experiment]
name = "GECCO_{TASK}"
output_dir = "results/gecco_final/{TASK}"

[grid]
algorithms = ["Standard_GA"]
tasks = ["{TASK}"]  # Single Task per file

# DIMENSIONS
dimensions = [20, 50]

# POPULATION SCALING (Alignment Stress Test)
pop_sizes = [
    128, 129,       # Small Scale
    1024, 1025,     # Neuroevolution Scale
    16384, 16385    # Hyperscale
]

# UNROLL FACTORS
unroll_factors = [1, 50]

# STATISTICS
repeats = 30
seeds = [42] # Master seed (expanded internally)
generations = 2000

[grid.hyperparams]
mutation_rate = 0.05
crossover_rate = 0.6
sigma = 0.1
elite_ratio = 0.1
"""

tasks = [
    "sphere",               # Group 1: Baseline
    "rosenbrock",           # Group 2: Valley
    "ellipsoidal_rotated",  # Group 3: Conditioning
    "rastrigin",            # Group 4: Multimodal
    "schaffers_f7"          # Group 5: Deceptive
]

os.makedirs("configs", exist_ok=True)

print("📦 Generating Checkpoint Configurations:")
for task in tasks:
    filename = f"configs/run_{task}.toml"
    content = base_toml.replace("{TASK}", task)
    
    with open(filename, "w") as f:
        f.write(content)
    
    print(f"   - Created {filename}")

print("\n✅ Done. You can now run them individually or via the master script.")