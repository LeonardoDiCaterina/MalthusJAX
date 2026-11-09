"""Independent routing-budget and rollout termination/anomaly contracts."""
import sys
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_publication_phase3 import adjacency, migration_routes
from publication_rl import InstrumentedRLEvaluator


def test_cluster_dry_run_never_uses_cpu_smoke_or_launches_workers(tmp_path):
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = tmp_path / "plan"
    subprocess.run([sys.executable, str(root / "scripts/run_publication_validation.py"),
                    "--dry-run", "--expect-revision", revision, "--output", str(output)],
                   cwd=root, check=True, capture_output=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stage"] == "cluster_validation_only"
    assert not manifest["publication_ready"]
    assert len(manifest["commands"]) == 3
    assert all("--smoke" not in command for command in manifest["commands"])
    assert manifest["completed"] == []
    assert not list(output.glob("phase*"))
    bad = subprocess.run([sys.executable, str(root / "scripts/run_publication_validation.py"),
                          "--dry-run", "--expect-revision", "wrong", "--output", str(tmp_path / "bad")],
                         cwd=root, capture_output=True)
    assert bad.returncode != 0
    assert not (tmp_path / "bad").exists()


def test_degree_does_not_multiply_migrant_budget():
    for topology in ("ring", "bidirectional", "fully_connected"):
        links = adjacency(topology, 4)
        traffic = {(i, j): 0 for i, targets in enumerate(links) for j in targets}
        for event in range(6):
            routes = migration_routes(links, 2, event)
            assert len(routes) == 8
            for island in range(4):
                sent = [r for r in routes if r["source"] == island]
                assert sorted(r["migrant_rank"] for r in sent) == [0, 1]
            for route in routes:
                assert route["source"] != route["target"]
                traffic[route["source"], route["target"]] += 1
        for island, targets in enumerate(links):
            counts = [traffic[island, j] for j in targets]
            assert max(counts) - min(counts) <= 1
    assert migration_routes(adjacency("single", 1), 2, 0) == []


@struct.dataclass
class Env:
    max_steps: int = struct.field(pytree_node=False, default=4)
    invalid: bool = struct.field(pytree_node=False, default=False)

    def reset(self, key):
        return jnp.array([0.0]), jnp.array(0)

    def preprocess_obs(self, obs):
        return obs

    def postprocess_action(self, action, obs):
        return action

    def step(self, state, action, key):
        next_state = state + 1
        reward = jnp.where(self.invalid & (state == 1), jnp.inf, 1.0)
        return jnp.array([next_state], dtype=float), next_state, reward, next_state >= 2, {}


@struct.dataclass
class Interpreter:
    def apply(self, genome, obs):
        return obs[0]


@struct.dataclass
class Output:
    def observe_step(self, state, obs, action, reward):
        return {"action": action}

    def process_rl(self, genome, reward, states, infos):
        valid = infos["valid_step"]
        mean_action = jnp.sum(jnp.where(valid, infos["action"], 0)) / jnp.maximum(jnp.sum(valid), 1)
        return -reward, {"descriptor": mean_action}


def test_terminal_padding_does_not_enter_descriptor():
    evaluator = InstrumentedRLEvaluator(env=Env(), interpreter=Interpreter(), output=Output())
    fitness, info = evaluator.evaluate(None, None, jax.random.PRNGKey(1))
    np.testing.assert_allclose(fitness, -2)
    np.testing.assert_allclose(info["descriptor"], .5)
    assert int(info["rollout_valid_steps"]) == 2
    assert int(info["rollout_terminations"]) == 1
    assert int(info["rollout_invalid_live_steps"]) == 0


def test_invalid_reward_is_reported_even_with_finite_fitness():
    evaluator = InstrumentedRLEvaluator(env=Env(invalid=True), interpreter=Interpreter(), output=Output())
    fitness, info = evaluator.evaluate(None, None, jax.random.PRNGKey(1))
    np.testing.assert_allclose(fitness, -1)
    assert int(info["rollout_live_steps"]) == 2
    assert int(info["rollout_valid_steps"]) == 1
    assert int(info["rollout_invalid_live_steps"]) == 1


def test_exploratory_schedule_is_separate_from_confirmation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    output = tmp_path / "explore"
    subprocess.run([sys.executable, str(root / "scripts/run_publication_validation.py"),
                    "--dry-run", "--stage", "exploratory", "--expect-revision", revision,
                    "--output", str(output)], check=True, capture_output=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stage"] == "exploratory"
    assert manifest["confirmation_eligible"] is False
    assert manifest["exploratory_protocol"]["seeds"] == [1000, 1001, 1002]
    assert manifest["exploratory_protocol"]["nominal_training_runs"] == {"1": 45, "2": 9, "3": 12}
    assert manifest["config_sha256"]
    assert manifest["completed"] == []
