"""Integration tests for the refactored evaluators and predictors."""

import pytest

from malthusjax.composer.composer import Composer

# Import LSP plugins so they register with the catalog
has_lsp = False
try:
    import lsp.evaluator.predictors  # noqa: F401
    has_lsp = True
except ImportError:
    pass


def test_gymnax_integration(tmp_path):
    """Test full pipeline execution for Gymnax with an MLP predictor."""
    toml_content = """
    [experiment]
    name = "integration_test"

    [pipelines.rl]
    engine_type = "ga"
    pop_size = 5
    num_generations = 2

    [pipelines.rl.genome]
    type = "real"
    shape = [1282]
    bounds = [-1.0, 1.0]

    [pipelines.rl.fitness]
    type = "RLEvaluator"

    [pipelines.rl.fitness.env]
    type = "GymnaxEnv"
    env_name = "CartPole-v1"

    [pipelines.rl.fitness.interpreter]
    type = "MLPInterpreter"
    input_dim = 4
    output_dim = 2
    hidden = [32, 32]
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    comparison = Composer.from_toml(str(toml_file))

    assert "rl" in comparison.pipelines
    assert len(comparison.pipelines["rl"].runs) > 0


@pytest.mark.skipif(not has_lsp, reason="LSP plugin not found")
def test_supervised_integration(tmp_path):
    """Test full pipeline execution for SupervisedEvaluator with a Linear predictor."""
    toml_content = """
    [experiment]
    name = "integration_test"

    [pipelines.supervised]
    engine_type = "ga"
    pop_size = 5
    num_generations = 2

    [pipelines.supervised.genome]
    type = "linear"
    length = 10
    num_inputs = 2

    [pipelines.supervised.fitness]
    type = "sklearn"
    dataset = "make_regression"
    n_samples = 10
    n_features = 2
    predictor = "linear"
    """
    toml_file = tmp_path / "exp.toml"
    toml_file.write_text(toml_content)

    comparison = Composer.from_toml(str(toml_file))

    assert "supervised" in comparison.pipelines
    assert len(comparison.pipelines["supervised"].runs) > 0
