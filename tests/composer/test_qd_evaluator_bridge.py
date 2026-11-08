"""Compliance and regression tests for QD evaluator bridge and population contracts."""

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from malthusjax.core.base import BasePopulation
from malthusjax.core.fitness.composable.base import (
    BaseDescriptorFn,
    GenotypeBoxDescriptorFn,
    IdentityTransform,
    QDOutput,
    RLActionSummaryDescriptorFn,
    RLFinalStateDescriptorFn,
    ScalarOutput,
)
from malthusjax.core.fitness.composable.environments import SphereEnv
from malthusjax.core.fitness.composable.evaluators import OptimizationEvaluator
from malthusjax.core.fitness.composable.interpreters import IdentityInterpreter
from malthusjax.core.fitness.qd.evaluator import (
    BaseQDEvaluator,
    ProjectedQDEvaluator,
    ensure_qd_evaluator,
    to_repertoire_fitness,
)
from malthusjax.core.genome.qd.population import QDPopulation, has_descriptors
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation


def test_base_population_purified_contract():
    """Verify BasePopulation does not expose leaky .descriptors property."""
    config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    genes = RealGenome(values=jnp.ones((4, 5)))
    pop = RealPopulation(genes=genes, fitness=jnp.zeros(4), config=config)

    assert not hasattr(pop, "descriptors")
    assert not has_descriptors(pop)


def test_qd_population_contract_and_upgrade():
    """Verify QDPopulation enforces non-null descriptors and allows clean upgrade."""
    config = RealGenomeConfig(shape=(5,), bounds=(-5.0, 5.0))
    genes = RealGenome(values=jnp.ones((4, 5)))
    pop = RealPopulation(
        genes=genes, fitness=jnp.zeros(4), config=config, info={"diagnostics": 123}
    )

    desc = jnp.full((4, 2), 0.5)
    qd_pop = QDPopulation.from_population(pop, descriptors=desc)

    assert isinstance(qd_pop, QDPopulation)
    assert hasattr(qd_pop, "descriptors")
    assert has_descriptors(qd_pop)
    assert jnp.allclose(qd_pop.descriptors, 0.5)
    assert qd_pop.info["diagnostics"] == 123

    # Missing descriptors raises KeyError
    bad_pop = QDPopulation(genes=genes, fitness=jnp.zeros(4), config=config, info={})
    with pytest.raises(KeyError, match="Descriptors not found in population info"):
        _ = bad_pop.descriptors


def test_projected_qd_evaluator_scalar_projection():
    """Verify ensure_qd_evaluator projects scalar evaluator to [0, 1]^k descriptors."""
    env = SphereEnv()
    evaluator = OptimizationEvaluator(
        env=env,
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),
    )

    qd_eval = ensure_qd_evaluator(evaluator, num_descriptors=2, bounds=(-5.0, 5.0))
    assert isinstance(qd_eval, ProjectedQDEvaluator)
    assert qd_eval.env is env
    assert qd_eval.output == evaluator.output

    config = RealGenomeConfig(shape=(6,), bounds=(-5.0, 5.0))
    key = jax.random.PRNGKey(42)
    pop = config.init_population(key, 8)

    evaluated_pop = qd_eval.evaluate_population(pop)
    assert isinstance(evaluated_pop, QDPopulation)
    assert has_descriptors(evaluated_pop)
    assert evaluated_pop.descriptors.shape == (8, 2)
    assert jnp.all(evaluated_pop.descriptors >= 0.0)
    assert jnp.all(evaluated_pop.descriptors <= 1.0)


@struct.dataclass
class CustomConstantDescriptorFn(BaseDescriptorFn):
    constant_val: float = 0.77

    def compute(self, genome, inputs, outputs):
        return jnp.full((2,), self.constant_val)

    def compute_rl(self, genome, final_state, step_infos):
        return jnp.full((2,), self.constant_val)


def test_nondestructive_qd_output_preservation():
    """Verify true behavioral descriptors from QDOutput are never overwritten by genotypes."""
    env = SphereEnv()
    custom_desc = CustomConstantDescriptorFn(constant_val=0.77)
    qd_output = QDOutput.create(
        maximize=True, num_descriptors=2, descriptor_fn=custom_desc
    )
    evaluator = OptimizationEvaluator(
        env=env,
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=qd_output,
    )

    qd_eval = ensure_qd_evaluator(evaluator, num_descriptors=2, bounds=(-5.0, 5.0))

    config = RealGenomeConfig(shape=(6,), bounds=(-5.0, 5.0))
    key = jax.random.PRNGKey(0)
    pop = config.init_population(key, 8)

    evaluated_pop = qd_eval.evaluate_population(pop)
    assert isinstance(evaluated_pop, QDPopulation)
    # The descriptors MUST be 0.77 from the custom descriptor function, NOT sliced genotypes!
    assert jnp.allclose(evaluated_pop.descriptors, 0.77)


def test_to_repertoire_fitness_sign_handling():
    """Verify to_repertoire_fitness converts lower-is-better MalthusJAX scores to higher-is-better."""
    # 1. Composable evaluator maximizing (reward = 150 -> format_fitness returns -150)
    eval_max = OptimizationEvaluator(
        env=SphereEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=True),
    )
    raw_reward = jnp.array([150.0, 200.0])
    eval_pop_fitness_max = -raw_reward  # lower-is-better
    rep_fit_max = to_repertoire_fitness(eval_max, eval_pop_fitness_max, engine_maximize=True)
    # Repertoire must receive +reward so it maximizes it!
    assert jnp.allclose(rep_fit_max, raw_reward)

    # 2. Composable evaluator minimizing (loss = 0.25 -> format_fitness returns 0.25)
    eval_min = OptimizationEvaluator(
        env=SphereEnv(),
        transform=IdentityTransform(),
        interpreter=IdentityInterpreter(),
        output=ScalarOutput(maximize=False),
    )
    raw_loss = jnp.array([0.25, 1.5])
    eval_pop_fitness_min = raw_loss
    rep_fit_min = to_repertoire_fitness(eval_min, eval_pop_fitness_min, engine_maximize=False)
    # Repertoire must receive -loss so maximizing repertoire minimizes loss!
    assert jnp.allclose(rep_fit_min, -raw_loss)


def test_concrete_descriptor_functions():
    """Test GenotypeBoxDescriptorFn, RLFinalStateDescriptorFn, and RLActionSummaryDescriptorFn."""
    genome = jnp.array([-5.0, 5.0, 0.0])

    # Genotype box: -5 -> 0.0, +5 -> 1.0
    box_fn = GenotypeBoxDescriptorFn(num_descriptors=2, bounds=(-5.0, 5.0))
    d_box = box_fn.compute(genome, None, genome)
    assert jnp.allclose(d_box, jnp.array([0.0, 1.0]))

    # RL final state
    state_mock = struct.dataclass(type("MockState", (), {"obs": jnp.array([0.0, 1.0, -1.0])}))()
    rl_state_fn = RLFinalStateDescriptorFn(indices=(0, 2), bounds=(-1.0, 1.0))
    d_rl = rl_state_fn.compute_rl(genome, state_mock, {})
    assert jnp.allclose(d_rl, jnp.array([0.5, 0.0]))

    # RL action summary
    step_infos = {"action": jnp.array([[0.0, 1.0], [0.0, 1.0]])}
    rl_act_fn = RLActionSummaryDescriptorFn(num_descriptors=2, bounds=(-1.0, 1.0))
    d_act = rl_act_fn.compute_rl(genome, None, step_infos)
    assert jnp.allclose(d_act, jnp.array([0.5, 1.0]))
