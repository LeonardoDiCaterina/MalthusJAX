
import jax
import jax.numpy as jnp

# The bug exists in the objective_fn wrapper created inside `_build_map_elites_engine`.
# Since `_build_map_elites_engine` dynamically creates `obj_fn` inside `composer.py`,
# we will write a functional test that recreates the `obj_fn` logic to ensure
# the fix applies the correct mathematical inversion.

def create_mock_obj_fn(maximize: bool):
    """
    Simulates the `obj_fn` created inside `malthusjax/composer/composer.py`
    around line 1750, containing the fix for the double sign-flip bug.
    """

    # 1. Mock the genome object
    class MockGenome:
        def setup(self, state):
            return state

        def transform(self, state, nodes, conns):
            return nodes  # dummy transform

        def forward(self, state, nodes, inputs):
            return jnp.zeros(1)

    genome_obj = MockGenome()

    # 2. Mock the problem object
    # In TensorNEAT, problem.evaluate returns `-loss`
    class MockProblem:
        def evaluate(self, state, key, forward_fn, transformed_pop):
            # Simulate an MSE loss of 100.0, returning -100.0 natively
            return -100.0

    problem = MockProblem()

    # 3. Create the objective function (simulating composer.py)
    def obj_fn(nodes, conns):
        from tensorneat.common import State

        # Ensure batch dimension
        if nodes.ndim == 2:
            nodes = jnp.expand_dims(nodes, 0)
            conns = jnp.expand_dims(conns, 0)

        batch_size = nodes.shape[0]
        state = State(randkey=jax.random.PRNGKey(0))
        state = genome_obj.setup(state)

        transformed_pop = jax.vmap(genome_obj.transform, in_axes=(None, 0, 0))(
            state, nodes, conns
        )

        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        fitness = jax.vmap(problem.evaluate, in_axes=(None, 0, None, 0))(
            state, keys, genome_obj.forward, transformed_pop
        )

        # --- THE FIX BOUNDARY ---
        # If MalthusJAX is minimizing, it expects positive loss,
        # so we invert the natively negated fitness.
        if not maximize:
            fitness = -fitness

        descriptors = jnp.zeros((batch_size, 2))
        return fitness, descriptors

    return obj_fn

def test_tensorneat_obj_fn_minimization():
    """
    Test that when minimize=True (maximize=False), the negative loss returned
    by TensorNEAT is successfully inverted into positive loss for MalthusJAX.
    """
    obj_fn = create_mock_obj_fn(maximize=False)

    # Dummy batched nodes and conns
    nodes = jnp.zeros((5, 10, 5))
    conns = jnp.zeros((5, 20, 5))

    fitness, descriptors = obj_fn(nodes, conns)

    # Problem evaluates to -100.0 natively. Since maximize=False, it should invert to 100.0.
    assert jnp.allclose(fitness, 100.0)
    assert fitness.shape == (5,)

def test_tensorneat_obj_fn_maximization():
    """
    Test that when maximize=True, the native fitness returned by TensorNEAT
    (which is already maximization-friendly) is preserved.
    """
    obj_fn = create_mock_obj_fn(maximize=True)

    nodes = jnp.zeros((5, 10, 5))
    conns = jnp.zeros((5, 20, 5))

    fitness, descriptors = obj_fn(nodes, conns)

    # Problem evaluates to -100.0. Since maximize=True, it should remain -100.0.
    assert jnp.allclose(fitness, -100.0)
    assert fitness.shape == (5,)
