import jax
import jax.numpy as jnp
import pytest
from malthus_gp.genomes.cartesian_genome import CartesianGenome, CartesianGenomeConfig
from malthusjax.testing.compliance import GenomeComplianceSuite


class TestCartesianGenome(GenomeComplianceSuite):
    @pytest.fixture
    def component(self):
        config = CartesianGenomeConfig(
            num_rows=1,
            num_cols=10,
            num_inputs=2,
            num_outputs=1,
            num_ops=4,
            max_arity=2,
            levels_back=-1,
        )
        rng = jax.random.PRNGKey(0)
        return CartesianGenome.random_init(rng, config=config)


def test_cartesian_levels_back_bounds():
    config = CartesianGenomeConfig(
        num_rows=2,
        num_cols=5,
        num_inputs=2,
        num_outputs=1,
        num_ops=4,
        max_arity=2,
        levels_back=2,
    )
    lo, hi = config._col_connection_bounds()
    assert lo[0] == 0
    assert hi[0] == 2  # col 0 cannot connect to self
    assert lo[2] == 0
    assert hi[2] == 4  # col 1 can connect to inputs and col 0
    assert lo[4] == 2  # col 2 (2 >= 2) cannot connect to inputs, only col 0..1


def test_cartesian_autocorrect():
    config = CartesianGenomeConfig(
        num_rows=1,
        num_cols=5,
        num_inputs=2,
        num_outputs=1,
        num_ops=4,
        max_arity=2,
        levels_back=1,
    )
    # create illegal args
    ops = jnp.array([0, 1, 2, 3, 0], dtype=jnp.int32)
    args = jnp.array([[99, 99], [0, 0], [1, 1], [2, 2], [3, 3]], dtype=jnp.int32)
    out_nodes = jnp.array([99], dtype=jnp.int32)

    genome = CartesianGenome(ops=ops, args=args, out_nodes=out_nodes)
    corrected = genome.autocorrect(config)

    # col 0 (node 2) can only connect to inputs 0..1
    assert corrected.args[0, 0] <= 1
    # output node cannot exceed total nodes
    assert corrected.out_nodes[0] < config.num_inputs + config.num_nodes


def test_cartesian_active_mask():
    # 2 inputs (0, 1), 3 nodes (2, 3, 4), 1 output pointing to node 4
    # node 4 connects to node 3 and input 0
    # node 3 connects to input 1 and input 0
    # node 2 is unused (neutral node)
    ops = jnp.array([0, 0, 0], dtype=jnp.int32)
    args = jnp.array([[0, 1], [0, 1], [3, 0]], dtype=jnp.int32)
    out_nodes = jnp.array([4], dtype=jnp.int32)

    genome = CartesianGenome(ops=ops, args=args, out_nodes=out_nodes)
    mask = genome.get_active_mask(num_inputs=2)

    assert mask[0]  # input 0 is used
    assert mask[1]  # input 1 is used
    assert not mask[2]  # node 2 is inactive (neutral!)
    assert mask[3]  # node 3 is active
    assert mask[4]  # node 4 is active
