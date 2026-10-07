"""Tests for NeatInterpreter compliance and network execution."""

import jax
import jax.numpy as jnp
import pytest

from malthus_neat.genomes.neat_genome import NeatGenome, NeatGenomeConfig
from malthus_neat.interpreters.neat_interpreter import NeatInterpreter
from malthus_neat.interpreters.visualizer import extract_network_graph, to_dot
from malthusjax.testing.compliance import InterpreterComplianceSuite


class TestNeatInterpreter(InterpreterComplianceSuite):
    @pytest.fixture
    def config(self):
        return NeatGenomeConfig(
            num_inputs=2,
            num_outputs=1,
            max_nodes=8,
            max_conns=16,
        )

    @pytest.fixture
    def component(self, config):
        return NeatInterpreter(genome_config=config)

    @pytest.fixture
    def mock_genome(self, config):
        key = jax.random.PRNGKey(42)
        return NeatGenome.random_init(key, config)

    @pytest.fixture
    def mock_inputs(self):
        return jnp.array([0.5, -0.5], dtype=jnp.float32)


def test_neat_interpreter_batch_evaluation():
    config = NeatGenomeConfig(num_inputs=2, num_outputs=1, max_nodes=8, max_conns=16)
    interpreter = NeatInterpreter(genome_config=config)
    key = jax.random.PRNGKey(101)
    genome = NeatGenome.random_init(key, config)

    X = jnp.linspace(-1.0, 1.0, 20).reshape(-1, 2)
    preds = interpreter.evaluate_batch(genome, X)

    assert preds.shape == (10, 1)
    assert jnp.all(jnp.isfinite(preds))


def test_neat_visualization_export():
    config = NeatGenomeConfig(num_inputs=2, num_outputs=1, max_nodes=6, max_conns=10)
    key = jax.random.PRNGKey(123)
    genome = NeatGenome.random_init(key, config)

    nodes, edges = extract_network_graph(genome, config)
    assert len(nodes) >= 3
    assert len(edges) >= 1

    dot_str = to_dot(genome, config)
    assert "digraph NEAT" in dot_str
    assert "->" in dot_str
