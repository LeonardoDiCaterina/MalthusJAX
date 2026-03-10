from pathlib import Path

from malthusjax.benchmarking.runner import BenchmarkRunner
from malthusjax.composer import Composer


def test_quick_run_uses_default_trace_dir(monkeypatch, tmp_path):
    """Composer.quick_run should pass a default trace_dir when none provided.

    We monkeypatch BenchmarkRunner.__init__ to capture the keyword argument
    so we can assert it was set to the expected path.
    """
    captured: dict = {}

    original_init = BenchmarkRunner.__init__

    def fake_init(self, *args, **kwargs):
        # save trace_dir for assertion then call real constructor
        captured['trace_dir'] = kwargs.get('trace_dir')
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(BenchmarkRunner, '__init__', fake_init)

    composer = Composer.create_default()
    # run a trivial experiment; stub engine is used because no operators
    result = composer.quick_run(seeds=(0,), experiment_name="foo", output_dir=tmp_path)

    assert captured.get('trace_dir') == Path('results/traces')
    # the experiment itself should still succeed
    assert result is not None
    assert result.runs


def test_quick_run_can_override_trace_dir(tmp_path):
    """Supplying trace_dir explicitly should override the default."""
    composer = Composer.create_default()
    custom = tmp_path / "mytraces"
    result = composer.quick_run(seeds=(0,), experiment_name="bar", output_dir=tmp_path, trace_dir=custom)
    # check that directory was actually created by the runner
    assert custom.exists()
    assert any(custom.iterdir())  # should contain at least one trace subfolder
