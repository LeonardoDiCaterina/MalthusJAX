import jax.numpy as jnp

from malthusjax.core.genome.cartesian_genome import CartesianGenome, CartesianGenomeConfig


def test_levels_back_constraint_respected():
    """Test that the mathematical lower bound accurately restricts node reach."""
    N = 2
    nr = 2
    nc = 5
    l_back = 2

    config = CartesianGenomeConfig(
        num_rows=nr,
        num_cols=nc,
        num_inputs=N,
        num_outputs=1,
        num_ops=2,
        max_arity=2,
        levels_back=l_back,
    )

    lo, hi = config._col_connection_bounds()

    # Check node in column 0 (indices 0, 1 -> absolute 2, 3)
    # Since 0 < l (0 < 2), it should be able to reach inputs (lo = 0)
    assert lo[0] == 0
    assert hi[0] == N + 0 * nr  # 2

    # Check node in column 1 (indices 2, 3 -> absolute 4, 5)
    # Since 1 < l (1 < 2), it should be able to reach inputs (lo = 0)
    assert lo[2] == 0
    assert hi[2] == N + 1 * nr  # 4

    # Check node in column 2 (indices 4, 5 -> absolute 6, 7)
    # Since 2 >= l (2 >= 2), mask kicks in!
    # Must only reach max(0, 2 - 2) = column 0. Absolute index = N + 0*nr = 2.
    assert lo[4] == 2
    assert hi[4] == N + 2 * nr  # 6

    # Check node in column 4 (indices 8, 9 -> absolute 10, 11)
    # Since 4 >= 2. Reach column 4 - 2 = 2. Absolute index = N + 2*nr = 6.
    assert lo[8] == 6
    assert hi[8] == N + 4 * nr  # 10


def test_autocorrect_clamps_invalid_mutations():
    """Test that autocorrect strictly clamps illegal topological mutations."""
    config = CartesianGenomeConfig(
        num_rows=2, num_cols=3, num_inputs=2, num_outputs=1, num_ops=2, max_arity=2, levels_back=1
    )

    # Total nodes = 6 (absolute indices 2 through 7)
    # col 0: nodes 2, 3.  Can reach 0..1  (lo=0, hi=2)
    # col 1: nodes 4, 5.  Can reach 2..3  (lo=2, hi=4)
    # col 2: nodes 6, 7.  Can reach 4..5  (lo=4, hi=6)

    # Inject an invalid genome that tries to connect forward or out-of-bounds backward
    args = jnp.array(
        [
            [10, 10],  # col 0 node 2: connects forward to 10! (invalid, max is 1)
            [0, 1],  # col 0 node 3: valid
            [0, 5],  # col 1 node 4: 0 is backward too far (min is 2). 5 is forward (max is 3).
            [2, 3],  # col 1 node 5: valid
            [1, 1],  # col 2 node 6: 1 is backward too far (min is 4).
            [4, 5],  # col 2 node 7: valid
        ]
    )

    ops = jnp.zeros(6, dtype=jnp.int32)
    out_nodes = jnp.zeros(1, dtype=jnp.int32)

    genome = CartesianGenome(ops=ops, args=args, out_nodes=out_nodes)
    corrected = genome.autocorrect(config)

    # Node 2 (col 0): args 10 -> clipped to hi-1 = 1
    assert jnp.all(corrected.args[0] == jnp.array([1, 1]))

    # Node 3: remains valid
    assert jnp.all(corrected.args[1] == jnp.array([0, 1]))

    # Node 4 (col 1): args [0, 5]. 0 is clipped to lo=2. 5 is clipped to hi-1=3.
    assert jnp.all(corrected.args[2] == jnp.array([2, 3]))

    # Node 5: remains valid
    assert jnp.all(corrected.args[3] == jnp.array([2, 3]))

    # Node 6 (col 2): args [1, 1]. clipped to lo=4.
    assert jnp.all(corrected.args[4] == jnp.array([4, 4]))
