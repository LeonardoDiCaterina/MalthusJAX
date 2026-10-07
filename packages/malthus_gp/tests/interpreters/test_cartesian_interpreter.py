import jax.numpy as jnp
from malthus_gp.genomes.cartesian_genome import CartesianGenome
from malthus_gp.interpreters.cartesian_interpreter import CartesianInterpreter


def test_cartesian_interpreter_handcrafted_graph():
    # 2 inputs (indices 0, 1)
    # Node 2: add (op 0) on inputs 0 and 1 -> x0 + x1
    # Node 3: mul (op 2) on node 2 and input 0 -> (x0 + x1) * x0
    # Output: points to node 3
    ops = jnp.array([0, 2], dtype=jnp.int32)
    args = jnp.array([[0, 1], [2, 0]], dtype=jnp.int32)
    out_nodes = jnp.array([3], dtype=jnp.int32)

    genome = CartesianGenome(ops=ops, args=args, out_nodes=out_nodes)
    interpreter = CartesianInterpreter(num_inputs=2, num_outputs=1)

    # Test single input: x0=2.0, x1=3.0 -> (2 + 3) * 2 = 10.0
    x_single = jnp.array([2.0, 3.0])
    y_single = interpreter.apply(genome, x_single)
    assert jnp.isclose(y_single[0], 10.0)

    # Test batch inputs:
    # row 0: (2, 3) -> 10.0
    # row 1: (1, 4) -> (1 + 4) * 1 = 5.0
    X_batch = jnp.array([[2.0, 3.0], [1.0, 4.0]])
    y_batch = interpreter.apply(genome, X_batch)
    assert jnp.allclose(y_batch.squeeze(), jnp.array([10.0, 5.0]))

    # Test formula extraction
    formula = interpreter.to_formula(genome, feature_names=["a", "b"])
    assert formula == "((a + b) * a)"
