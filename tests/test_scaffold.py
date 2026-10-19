"""Unit and integration tests for scripts/scaffold.py component scaffolder."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.scaffold as scaffold_cli


def _to_snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


COMPONENT_TEST_MATRIX = [
    ("mutation", "CustomMutation", "custom_mut", None),
    ("crossover", "CustomCrossover", "custom_cross", None),
    ("selection", "CustomSelection", "custom_sel", None),
    ("emitter", "CustomEmitter", "custom_emit", None),
    ("engine", "CustomEngine", "custom_engine", None),
    ("evaluator", "CustomEvaluator", None, None),
    ("transform", "CustomTransform", "custom_transform", None),
    ("genome", "CustomGenome", "custom_genome", None),
    ("population", "CustomPopulation", "custom_population", None),
    ("adapter", "CustomAdapter", "custom_adapter", None),
    ("interpreter", "CustomInterpreter", "custom_interpreter", None),
    ("environment", "CustomOptEnv", "custom_opt_env", "opt"),
    ("environment", "CustomSupEnv", "custom_sup_env", "supervised"),
    ("environment", "CustomRLEnv", "custom_rl_env", "rl"),
    ("backend", "CustomBackend", "custom_backend", None),
]


@pytest.mark.parametrize("comp_type,name,key,env_type", COMPONENT_TEST_MATRIX)
def test_scaffold_component_e2e(
    tmp_path: Path,
    comp_type: str,
    name: str,
    key: str | None,
    env_type: str | None,
):
    """Scaffold every supported component type and verify generated compliance tests pass."""
    impl_dir = tmp_path / "plugins"
    test_dir = tmp_path / "tests" / "plugins"

    args = [
        "-t",
        comp_type,
        "-n",
        name,
        "-o",
        str(impl_dir),
        "-to",
        str(test_dir),
    ]
    if key is not None:
        args.extend(["-k", key])
    if env_type is not None:
        args.extend(["--env-type", env_type])

    # Run scaffolder via CLI entry point
    ret = scaffold_cli.main(args)
    assert ret == 0

    module_name = _to_snake_case(name)
    impl_file = impl_dir / f"{module_name}.py"
    test_file = test_dir / f"test_{module_name}.py"

    assert impl_file.exists(), f"Implementation file was not created: {impl_file}"
    assert test_file.exists(), f"Test file was not created: {test_file}"

    # Verify generated test file executes and passes compliance suite cleanly
    env = os.environ.copy()
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = f"{tmp_path}:{src_dir}:" + env.get("PYTHONPATH", "")

    test_proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "--no-cov", "-q"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert test_proc.returncode == 0, (
        f"Generated test failed for {comp_type} ({name}):\n"
        f"STDOUT:\n{test_proc.stdout}\n"
        f"STDERR:\n{test_proc.stderr}"
    )


def test_scaffold_missing_key_error(tmp_path: Path):
    """Verify scaffolder fails when --key is missing for non-evaluator components."""
    args = [
        "-t",
        "mutation",
        "-n",
        "NoKeyMutation",
        "-o",
        str(tmp_path / "plugins"),
        "-to",
        str(tmp_path / "tests"),
    ]
    ret = scaffold_cli.main(args)
    assert ret == 1


def test_scaffold_already_exists_error(tmp_path: Path):
    """Verify scaffolder refuses to overwrite an existing file without error."""
    impl_dir = tmp_path / "plugins"
    test_dir = tmp_path / "tests"

    args = [
        "-t",
        "evaluator",
        "-n",
        "ExistingEvaluator",
        "-o",
        str(impl_dir),
        "-to",
        str(test_dir),
    ]

    ret1 = scaffold_cli.main(args)
    assert ret1 == 0

    # Second invocation should error because files already exist
    ret2 = scaffold_cli.main(args)
    assert ret2 == 1
