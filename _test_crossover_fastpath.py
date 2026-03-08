import jax
from malthusjax.core.genome.binary_genome import BinaryGenomeConfig, BinaryPopulation
from malthusjax.operators.crossover.binary import UniformCrossover, SinglePointCrossover

key = jax.random.PRNGKey(0)
config = BinaryGenomeConfig(shape=(8,))
pop1 = BinaryPopulation.init_random(jax.random.PRNGKey(1), config, size=4)
pop2 = BinaryPopulation.init_random(jax.random.PRNGKey(2), config, size=4)

# K=1 fast path
op = UniformCrossover(num_offspring=1, input_length=4, typed_keys=False)
keys = jax.random.split(key, op.num_keys((4,)))
out1 = op(keys, pop1, pop2, config)
assert out1.genes.values.shape == (4, 8), f"K=1 shape wrong: {out1.genes.values.shape}"

# K=2 nested path
op2 = UniformCrossover(num_offspring=2, input_length=4, typed_keys=False)
keys2 = jax.random.split(key, op2.num_keys((4,)))
out2 = op2(keys2, pop1, pop2, config)
assert out2.genes.values.shape == (8, 8), f"K=2 shape wrong: {out2.genes.values.shape}"

# SinglePoint K=1
op3 = SinglePointCrossover(num_offspring=1, input_length=4, typed_keys=False)
keys3 = jax.random.split(key, op3.num_keys((4,)))
out3 = op3(keys3, pop1, pop2, config)
assert out3.genes.values.shape == (4, 8), f"SinglePoint K=1 shape wrong: {out3.genes.values.shape}"

print(f"K=1: {out1.genes.values.shape}  K=2: {out2.genes.values.shape}  SinglePoint K=1: {out3.genes.values.shape}")
print("All OK")
