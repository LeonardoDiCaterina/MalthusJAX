import jax
import jax.numpy as jnp
from sklearn.datasets import fetch_california_housing, load_diabetes, load_breast_cancer, make_regression, make_classification

from malthusjax.core.fitness.linear_gp_evaluator import LinearGPEvaluator, LinearGPEvaluatorConfig
from lsp.evaluator.cartesian import CartesianGPEvaluator, CartesianGPEvaluatorConfig
from lsp.evaluator.neural_linear import NeuralPrefixEvaluator, NeuralPrefixEvaluatorConfig
from lsp.evaluator.neural_cartesian import NeuralCartesianEvaluator, NeuralCartesianEvaluatorConfig
from lsp.evaluator.differentiable_evaluator import DifferentiableCartesianGPEvaluator, DifferentiableCartesianGPEvaluatorConfig

from malthusjax.composer.decorators import register_fitness


def _get_sklearn_data(dataset_name: str, **kwargs):
    """Fetches and JAX-ifies scikit-learn datasets."""
    # Pop 'seed' injected by Composer to use as random_state
    seed = kwargs.pop("seed", 42)
    if isinstance(seed, str):
        seed = int(seed)

    if dataset_name == "california_housing":
        data = fetch_california_housing()
    elif dataset_name == "diabetes":
        data = load_diabetes()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    elif dataset_name == "make_regression":
        X, y = make_regression(random_state=seed, **kwargs)
        return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
    elif dataset_name == "make_classification":
        X, y = make_classification(random_state=seed, **kwargs)
        return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    
    return jnp.array(data.data, dtype=jnp.float32), jnp.array(data.target, dtype=jnp.float32)


# =============================================================================
# CGP Evaluators
# =============================================================================

@register_fitness(name="sklearn_cgp")
class SklearnCGPEvaluator(CartesianGPEvaluator):
    def __init__(self, dataset: str, config: CartesianGPEvaluatorConfig = None, **kwargs):
        X, y = _get_sklearn_data(dataset, **kwargs)
        if config is None:
            config = CartesianGPEvaluatorConfig(batch_size=X.shape[0])
        else:
            config = config.replace(batch_size=X.shape[0])
        super().__init__(config=config, data=(X, y))


@register_fitness(name="equation_cgp")
class EquationCGPEvaluator(CartesianGPEvaluator):
    def __init__(self, equation_name: str, num_samples: int = 1000, key_seed: int = 42, config: CartesianGPEvaluatorConfig = None):
        key = jax.random.PRNGKey(key_seed)
        
        if equation_name == "feynman_01":
            # y = x_0^2 + sin(x_1)
            X = jax.random.uniform(key, (num_samples, 2), minval=-5.0, maxval=5.0)
            y = X[:, 0]**2 + jnp.sin(X[:, 1])
        else:
            raise ValueError(f"Unknown equation {equation_name}")
        
        if config is None:
            config = CartesianGPEvaluatorConfig(batch_size=X.shape[0])
        else:
            config = config.replace(batch_size=X.shape[0])
            
        super().__init__(config=config, data=(X, y))


# =============================================================================
# MEP Evaluators
# =============================================================================

@register_fitness(name="sklearn_mep")
class SklearnMEPEvaluator(LinearGPEvaluator):
    def __init__(self, dataset: str, config: LinearGPEvaluatorConfig = None, **kwargs):
        X, y = _get_sklearn_data(dataset, **kwargs)
        if config is None:
            config = LinearGPEvaluatorConfig(num_inputs=X.shape[1])
        else:
            config = config.replace(num_inputs=X.shape[1])
        super().__init__(config=config, data=(X, y))


@register_fitness(name="equation_mep")
class EquationMEPEvaluator(LinearGPEvaluator):
    def __init__(self, equation_name: str, num_samples: int = 1000, key_seed: int = 42, config: LinearGPEvaluatorConfig = None):
        key = jax.random.PRNGKey(key_seed)
        
        if equation_name == "feynman_01":
            # y = x_0^2 + sin(x_1)
            X = jax.random.uniform(key, (num_samples, 2), minval=-5.0, maxval=5.0)
            y = X[:, 0]**2 + jnp.sin(X[:, 1])
        else:
            raise ValueError(f"Unknown equation {equation_name}")
        
        if config is None:
            config = LinearGPEvaluatorConfig()
            
        super().__init__(config=config, data=(X, y))


# =============================================================================
# dMEP Evaluators
# =============================================================================

@register_fitness(name="sklearn_dmep")
class SklearnDMEPEvaluator(NeuralPrefixEvaluator):
    def __init__(self, dataset: str, config: NeuralPrefixEvaluatorConfig = None, **kwargs):
        X, y = _get_sklearn_data(dataset, **kwargs)
        if config is None:
            config = NeuralPrefixEvaluatorConfig(num_inputs=X.shape[1], length=100)
        super().__init__(config=config, data=(X, y))


@register_fitness(name="equation_dmep")
class EquationDMEPEvaluator(NeuralPrefixEvaluator):
    def __init__(self, equation_name: str, num_samples: int = 1000, key_seed: int = 42, config: NeuralPrefixEvaluatorConfig = None):
        key = jax.random.PRNGKey(key_seed)
        
        if equation_name == "feynman_01":
            # y = x_0^2 + sin(x_1)
            X = jax.random.uniform(key, (num_samples, 2), minval=-5.0, maxval=5.0)
            y = X[:, 0]**2 + jnp.sin(X[:, 1])
        else:
            raise ValueError(f"Unknown equation {equation_name}")
        
        if config is None:
            config = NeuralPrefixEvaluatorConfig(num_inputs=X.shape[1], length=100)
            
        super().__init__(config=config, data=(X, y))

@register_fitness(name="sklearn_neural_cgp")
class SklearnNeuralCGPEvaluator(NeuralCartesianEvaluator):
    def __init__(self, dataset: str, config: NeuralCartesianEvaluatorConfig = None, **kwargs):
        X, y = _get_sklearn_data(dataset, **kwargs)
        if config is None:
            config = NeuralCartesianEvaluatorConfig(batch_size=X.shape[0])
        else:
            config = config.replace(batch_size=X.shape[0])
        super().__init__(config=config, data=(X, y))

@register_fitness(name="sklearn_dcgp")
class SklearnDifferentiableCGPEvaluator(DifferentiableCartesianGPEvaluator):
    def __init__(self, dataset: str, config: DifferentiableCartesianGPEvaluatorConfig = None, **kwargs):
        X, y = _get_sklearn_data(dataset, **kwargs)
        if config is None:
            config = DifferentiableCartesianGPEvaluatorConfig(batch_size=X.shape[0], grad_weight=0.0)
        else:
            config = config.replace(batch_size=X.shape[0], grad_weight=0.0)
        # Pass dummy dY_dX since make_regression doesn't have it
        dY_dX = jnp.zeros_like(X)
        super().__init__(config=config, data=(X, y, dY_dX))
