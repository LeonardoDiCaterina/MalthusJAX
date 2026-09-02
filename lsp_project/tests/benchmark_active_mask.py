import time
import jax
import jax.numpy as jnp
from lsp.operators.cartesian_crossover import get_active_mask_scan, get_active_mask_matrix

def main():
    # Setup test case
    num_inputs = 5
    num_nodes = 1000
    max_arity = 2
    batch_size = 100
    
    # Generate random DAG (ensure validity: node i can only connect to j < i + num_inputs)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    # Generate args for one genome
    # For node i (0 to num_nodes-1), it can connect to [0, num_inputs + i - 1]
    def gen_args(i):
        # We just generate random args in valid range
        # Note: this is a static mock, real genomes would have valid args
        return jax.random.randint(jax.random.fold_in(subkey, i), (max_arity,), minval=0, maxval=num_inputs + i)
    
    args = jax.vmap(gen_args)(jnp.arange(num_nodes))
    out_nodes = jax.random.randint(key, (3,), minval=0, maxval=num_inputs + num_nodes)
    
    # Expand to batch
    args_batch = jnp.repeat(args[None, ...], batch_size, axis=0)
    out_nodes_batch = jnp.repeat(out_nodes[None, ...], batch_size, axis=0)
    
    print(f"Benchmarking with num_nodes={num_nodes}, batch_size={batch_size}")
    
    # JIT compile scan version
    scan_fn = jax.jit(jax.vmap(get_active_mask_scan, in_axes=(0, 0, None, None)), static_argnums=(2, 3))
    
    # JIT compile matrix version
    matrix_fn = jax.jit(jax.vmap(get_active_mask_matrix, in_axes=(0, 0, None, None)), static_argnums=(2, 3))
    
    # Warmup
    _ = scan_fn(args_batch, out_nodes_batch, num_inputs, num_nodes)
    _ = matrix_fn(args_batch, out_nodes_batch, num_inputs, num_nodes)
    
    # Verify they output the same thing
    scan_res = scan_fn(args_batch, out_nodes_batch, num_inputs, num_nodes)
    matrix_res = matrix_fn(args_batch, out_nodes_batch, num_inputs, num_nodes)
    assert jnp.all(scan_res == matrix_res), "Results do not match!"
    print("Correctness verified: Both approaches yield identical active masks.")
    
    # Timing Scan
    start = time.time()
    for _ in range(100):
        _ = scan_fn(args_batch, out_nodes_batch, num_inputs, num_nodes).block_until_ready()
    scan_time = (time.time() - start) / 100
    print(f"Scan Approach Time:   {scan_time * 1000:.3f} ms / batch")
    
    # Timing Matrix
    start = time.time()
    for _ in range(100):
        _ = matrix_fn(args_batch, out_nodes_batch, num_inputs, num_nodes).block_until_ready()
    matrix_time = (time.time() - start) / 100
    print(f"Matrix Approach Time: {matrix_time * 1000:.3f} ms / batch")

if __name__ == "__main__":
    main()
