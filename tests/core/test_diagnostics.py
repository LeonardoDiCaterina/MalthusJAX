"""Unit and integration tests for malthusjax.core.diagnostics."""

import io
import os
import signal
from unittest.mock import MagicMock, patch

from malthusjax.core.diagnostics import (
    _native_crash_signal_handler,
    format_crash_banner,
    get_environment_diagnostics,
    install_crash_handler,
    print_environment_diagnostics,
    stabilize_runtime_environment,
    uninstall_crash_handler,
)


def test_stabilize_runtime_environment_defaults():
    """Verify that stabilize_runtime_environment sets safe defaults when unset."""
    with patch.dict(os.environ, {}, clear=True):
        stabilize_runtime_environment()
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
        assert os.environ.get("NUMEXPR_NUM_THREADS") == "1"
        assert os.environ.get("VECLIB_MAXIMUM_THREADS") == "1"
        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"


def test_stabilize_runtime_environment_preserves_user_settings():
    """Verify that pre-existing user environment variables are never overwritten."""
    custom_env = {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "4",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }
    with patch.dict(os.environ, custom_env, clear=True):
        stabilize_runtime_environment()
        assert os.environ.get("OMP_NUM_THREADS") == "8"
        assert os.environ.get("MKL_NUM_THREADS") == "4"
        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true"
        # Others that were unset are still set
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"


def test_stabilize_runtime_environment_opt_out():
    """Verify that MALTHUSJAX_DISABLE_ENV_STABILIZATION skips environment clamping."""
    with patch.dict(os.environ, {"MALTHUSJAX_DISABLE_ENV_STABILIZATION": "1"}, clear=True):
        stabilize_runtime_environment()
        assert "OMP_NUM_THREADS" not in os.environ
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ


def test_format_crash_banner():
    """Verify that the crash banner formats signal details and remediation guidance."""
    banner = format_crash_banner(signal.SIGSEGV)
    assert "SIGSEGV" in banner
    assert "MalthusJAX Crash Diagnostic Reporter" in banner
    assert "CPU Cores Detected:" in banner
    assert "OMP_NUM_THREADS:" in banner
    assert "XLA Preallocation:" in banner
    assert "Most Common Causes & Resolutions:" in banner
    assert "https://github.com/LeonardoDiCaterina/MalthusJAX/issues" in banner


def test_install_and_uninstall_crash_handler():
    """Verify that crash handler registers and cleanly unregisters signals."""
    uninstall_crash_handler()
    install_crash_handler(enable_fault_handler=False)

    curr_handler = signal.getsignal(signal.SIGSEGV)
    assert curr_handler == _native_crash_signal_handler

    uninstall_crash_handler()
    restored_handler = signal.getsignal(signal.SIGSEGV)
    assert restored_handler != _native_crash_signal_handler


def test_install_crash_handler_opt_out():
    """Verify that MALTHUSJAX_DISABLE_CRASH_HANDLER prevents installation."""
    uninstall_crash_handler()
    with patch.dict(os.environ, {"MALTHUSJAX_DISABLE_CRASH_HANDLER": "1"}):
        install_crash_handler(enable_fault_handler=False)
        assert signal.getsignal(signal.SIGSEGV) != _native_crash_signal_handler
    uninstall_crash_handler()


def test_get_environment_diagnostics():
    """Verify that get_environment_diagnostics returns complete system info."""
    diag = get_environment_diagnostics()
    assert "platform" in diag
    assert "python_version" in diag
    assert "cpu_count" in diag
    assert "omp_num_threads" in diag
    assert "warnings" in diag
    assert isinstance(diag["warnings"], list)


def test_get_environment_diagnostics_warnings():
    """Verify warning generation under high core count simulation."""
    with patch("os.cpu_count", return_value=64):
        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "64",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            },
            clear=True,
        ):
            with patch("sys.platform", "linux"):
                diag = get_environment_diagnostics()
                assert len(diag["warnings"]) >= 1
                assert any("OMP_NUM_THREADS" in w for w in diag["warnings"])
                assert any("XLA_PYTHON_CLIENT_PREALLOCATE" in w for w in diag["warnings"])


def test_print_environment_diagnostics():
    """Verify print_environment_diagnostics formats output to stream."""
    stream = io.StringIO()
    print_environment_diagnostics(stream=stream)
    output = stream.getvalue()
    assert "MalthusJAX Environment & HPC Diagnostics" in output
    assert "Platform / Python:" in output
    assert "CPU Cores Detected:" in output


def test_native_crash_signal_handler_execution():
    """Verify that _native_crash_signal_handler flushes logs, outputs banner, and exits."""
    mock_exit = MagicMock()
    mock_stderr = io.StringIO()

    with patch("logging.shutdown") as mock_shutdown:
        with patch("sys.stderr", mock_stderr):
            with patch("os._exit", mock_exit):
                with patch("faulthandler.dump_traceback") as mock_trace:
                    _native_crash_signal_handler(signal.SIGSEGV, None)

                    assert mock_shutdown.called
                    assert mock_trace.called
                    mock_exit.assert_called_once_with(128 + signal.SIGSEGV)
                    assert "SIGSEGV" in mock_stderr.getvalue()


def test_native_crash_signal_handler_chains_to_previous():
    """Verify that _native_crash_signal_handler delegates to a previous custom handler."""
    mock_exit = MagicMock()
    mock_prev_handler = MagicMock()

    with patch(
        "malthusjax.core.diagnostics._PREVIOUS_HANDLERS", {signal.SIGSEGV: mock_prev_handler}
    ):
        with patch("logging.shutdown"):
            with patch("sys.stderr", io.StringIO()):
                with patch("os._exit", mock_exit):
                    with patch("faulthandler.dump_traceback"):
                        _native_crash_signal_handler(signal.SIGSEGV, None)

                        assert mock_prev_handler.called
                        # If chained handler returned, os._exit should not have been called
                        assert not mock_exit.called
