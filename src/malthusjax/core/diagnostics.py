"""Runtime process stabilization, environment sanitization, and native crash diagnostics.

This module provides Level 1 foundational utilities to safeguard MalthusJAX against
native C/C++ segmentation faults, OpenMP/BLAS thread contention on high-core HPC nodes,
CUDA driver context deadlocks, and silent process crashes.
"""

from __future__ import annotations

import faulthandler
import logging
import multiprocessing as mp
import os
import signal
import sys
from typing import Any, Dict, List, Optional, TextIO

_CRASH_HANDLER_INSTALLED: bool = False
_PREVIOUS_HANDLERS: Dict[int, Any] = {}

# Signal names mapping for readable diagnostics
_SIGNAL_NAMES: Dict[int, str] = {
    signal.SIGSEGV: "SIGSEGV (Segmentation Fault - Invalid Memory Access)",
}
if hasattr(signal, "SIGBUS"):
    _SIGNAL_NAMES[signal.SIGBUS] = "SIGBUS (Bus Error - Unaligned or Non-Existent Physical Memory)"
if hasattr(signal, "SIGFPE"):
    _SIGNAL_NAMES[signal.SIGFPE] = "SIGFPE (Fatal Floating-Point Exception)"
if hasattr(signal, "SIGABRT"):
    _SIGNAL_NAMES[signal.SIGABRT] = "SIGABRT (Process Aborted by C Library or Assertion)"


def stabilize_runtime_environment() -> None:
    """Stabilizes the runtime environment with safe defaults for OpenMP, BLAS, and JAX.

    Respects pre-existing user environment variables: `os.environ.setdefault` is used
    exclusively, ensuring that any explicit configuration set by the user or cluster
    job scheduler is 100% preserved.

    Can be completely disabled by setting `MALTHUSJAX_DISABLE_ENV_STABILIZATION=1`.
    """
    if os.environ.get("MALTHUSJAX_DISABLE_ENV_STABILIZATION", "").lower() in ("1", "true", "yes"):
        return

    # 1. Clamp C/Fortran math thread pools to prevent thread explosion and OpenMP
    # thread pool race conditions on high-core Linux nodes (e.g. 64-128 core HPC nodes).
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")

    # 2. Prevent JAX/XLA from locking 90% of GPU VRAM on initialization.
    # Enables multi-pipeline execution, side-by-side benchmarking, and shared GPU cluster usage.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # 3. Enforce safe multiprocessing start method on POSIX systems.
    # Avoids inheriting active CUDA contexts or OpenMP lock structures across fork().
    if sys.platform != "win32":
        try:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn", force=False)
        except (RuntimeError, ValueError):
            pass


def format_crash_banner(signum: int) -> str:
    """Formats an actionable, human-readable diagnostic banner for a native crash signal."""
    sig_desc = _SIGNAL_NAMES.get(signum, f"Signal {signum}")
    cpu_count = os.cpu_count() or 1
    omp_val = os.environ.get("OMP_NUM_THREADS", "<unset - defaulting to all cores>")
    mkl_val = os.environ.get("MKL_NUM_THREADS", "<unset>")
    openblas_val = os.environ.get("OPENBLAS_NUM_THREADS", "<unset>")
    cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "<all devices visible>")
    xla_prealloc = os.environ.get(
        "XLA_PYTHON_CLIENT_PREALLOCATE", "<default (true/preallocate 90%)>"
    )
    mp_method = mp.get_start_method(allow_none=True) or "default"

    lines = [
        "=" * 80,
        f"🚨 [MalthusJAX Crash Diagnostic Reporter] Fatal Native Crash Intercepted: {sig_desc}",
        "=" * 80,
        "A fatal hardware or C-level error occurred outside Python exception handling.",
        "",
        "Active System Diagnostics:",
        f"  • Platform / Python:   {sys.platform} | Python {sys.version.split()[0]}",
        f"  • CPU Cores Detected:  {cpu_count}",
        f"  • OMP_NUM_THREADS:     {omp_val}",
        f"  • MKL_NUM_THREADS:     {mkl_val}",
        f"  • OPENBLAS_NUM_THREADS:{openblas_val}",
        f"  • CUDA_VISIBLE_DEVICES:{cuda_devs}",
        f"  • XLA Preallocation:   {xla_prealloc}",
        f"  • MP Start Method:     {mp_method}",
        "",
        "Most Common Causes & Resolutions:",
        "  1. OpenMP / BLAS Thread Contention on High-Core Linux Servers:",
        "     If running on a multi-core machine (>= 16 cores), C extensions (KMeans, cKDTree,",
        "     BLAS) default to spawning threads across ALL CPU cores, conflicting with JAX.",
        "     👉 Fix: export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        "",
        "  2. GPU Memory (VRAM) Out of Memory / Preallocation Collision:",
        "     By default, JAX locks up to 90% of GPU memory at startup. When multiple pipelines,",
        "     processes, or frameworks run simultaneously on the same GPU, CUDA terminates.",
        "     👉 Fix: export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "",
        "  3. Process Forking with Active CUDA / JAX Runtime Handles:",
        "     Calling fork() in a process that already initialized CUDA causes driver segfaults.",
        "     👉 Fix: Ensure multiprocessing uses 'spawn' or launch jobs via the 'mjax' CLI.",
        "",
        "For troubleshooting or reporting bugs, please include the trace below at:",
        "  https://github.com/LeonardoDiCaterina/MalthusJAX/issues",
        "=" * 80,
        "CPython Thread Stack Trace at Fault Site:",
        "",
    ]
    return "\n".join(lines)


def _native_crash_signal_handler(signum: int, frame: Any) -> None:
    """Low-level signal handler invoked on SIGSEGV, SIGBUS, SIGFPE, or SIGABRT."""
    # 1. Flush all active loggers so in-flight experiment logs are committed to disk
    try:
        logging.shutdown()
    except Exception:
        pass

    # 2. Write diagnostic banner directly to stderr using low-level unbuffered write
    try:
        banner = format_crash_banner(signum)
        sys.stderr.write(banner)
        sys.stderr.flush()
    except Exception:
        pass

    # 3. Dump CPython thread stack traces
    try:
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    except Exception:
        pass

    # 4. Chain to previous handler if one was registered and is callable
    prev = _PREVIOUS_HANDLERS.get(signum)
    if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
        try:
            prev(signum, frame)
            return
        except Exception:
            pass

    # 5. Cleanly terminate with standard POSIX signal exit code (128 + signal_number)
    os._exit(128 + signum)


def install_crash_handler(enable_fault_handler: bool = True) -> None:
    """Installs POSIX signal handlers and enables `faulthandler` for crash diagnostics.

    Hooks `SIGSEGV`, `SIGBUS`, `SIGFPE`, and `SIGABRT` to output an actionable,
    formatted diagnostic report before process termination.

    Can be disabled by setting `MALTHUSJAX_DISABLE_CRASH_HANDLER=1`.
    """
    global _CRASH_HANDLER_INSTALLED
    if _CRASH_HANDLER_INSTALLED:
        return

    if os.environ.get("MALTHUSJAX_DISABLE_CRASH_HANDLER", "").lower() in ("1", "true", "yes"):
        return

    # Enable native CPython faulthandler
    if enable_fault_handler:
        try:
            faulthandler.enable(file=sys.stderr, all_threads=True)
        except Exception:
            pass

    # Trap critical native fault signals
    signals_to_trap: List[int] = [signal.SIGSEGV]
    if hasattr(signal, "SIGBUS"):
        signals_to_trap.append(signal.SIGBUS)
    if hasattr(signal, "SIGFPE"):
        signals_to_trap.append(signal.SIGFPE)
    if hasattr(signal, "SIGABRT"):
        signals_to_trap.append(signal.SIGABRT)

    for sig in signals_to_trap:
        try:
            prev = signal.getsignal(sig)
            if prev != _native_crash_signal_handler:
                _PREVIOUS_HANDLERS[sig] = prev
                signal.signal(sig, _native_crash_signal_handler)
        except (ValueError, OSError, AttributeError):
            # Signal handling might not be permitted in non-main threads or restricted runtimes
            pass

    _CRASH_HANDLER_INSTALLED = True


def uninstall_crash_handler() -> None:
    """Restores original signal handlers and disables crash interception."""
    global _CRASH_HANDLER_INSTALLED
    for sig, handler in list(_PREVIOUS_HANDLERS.items()):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, AttributeError):
            pass
    _PREVIOUS_HANDLERS.clear()
    _CRASH_HANDLER_INSTALLED = False


def get_environment_diagnostics() -> Dict[str, Any]:
    """Inspects the active runtime and returns a structured health diagnostics dictionary."""
    cpu_count = os.cpu_count() or 1
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    mkl_threads = os.environ.get("MKL_NUM_THREADS")
    openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS")
    xla_prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES")

    warnings: List[str] = []
    if sys.platform.startswith("linux") and cpu_count >= 16:
        if omp_threads is None or (omp_threads.isdigit() and int(omp_threads) > 4):
            warnings.append(
                f"High CPU core count ({cpu_count}) with unconstrained OMP_NUM_THREADS={omp_threads}. "
                "OpenMP may collide with JAX; recommend setting OMP_NUM_THREADS=1."
            )
    if xla_prealloc is None or xla_prealloc.lower() in ("true", "1"):
        warnings.append(
            "XLA_PYTHON_CLIENT_PREALLOCATE is active or unset. "
            "JAX will preallocate 75-90% of GPU VRAM; recommend setting to 'false' for multi-pipeline runs."
        )

    return {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "cpu_count": cpu_count,
        "omp_num_threads": omp_threads,
        "mkl_num_threads": mkl_threads,
        "openblas_num_threads": openblas_threads,
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "xla_python_client_preallocate": xla_prealloc,
        "cuda_visible_devices": cuda_devs,
        "multiprocessing_start_method": mp.get_start_method(allow_none=True),
        "crash_handler_installed": _CRASH_HANDLER_INSTALLED,
        "warnings": warnings,
    }


def print_environment_diagnostics(stream: Optional[TextIO] = None) -> None:
    """Formats and prints an environment health card to the given output stream."""
    out = stream if stream is not None else sys.stdout
    diag = get_environment_diagnostics()

    status_icon = "⚠️" if diag["warnings"] else "✅"
    lines = [
        "=" * 70,
        f"{status_icon} MalthusJAX Environment & HPC Diagnostics",
        "=" * 70,
        f"Platform / Python:    {diag['platform']} (Python {diag['python_version']})",
        f"CPU Cores Detected:   {diag['cpu_count']}",
        f"OMP_NUM_THREADS:      {diag['omp_num_threads'] or '<unset>'}",
        f"MKL_NUM_THREADS:      {diag['mkl_num_threads'] or '<unset>'}",
        f"OPENBLAS_NUM_THREADS: {diag['openblas_num_threads'] or '<unset>'}",
        f"XLA Preallocation:    {diag['xla_python_client_preallocate'] or '<default>'}",
        f"CUDA_VISIBLE_DEVICES: {diag['cuda_visible_devices'] or '<all>'}",
        f"Multiprocessing:      {diag['multiprocessing_start_method'] or '<default>'}",
        f"Crash Handler:        {'Active' if diag['crash_handler_installed'] else 'Inactive'}",
    ]

    if diag["warnings"]:
        lines.append("-" * 70)
        lines.append("Warnings & Recommendations:")
        for w in diag["warnings"]:
            lines.append(f"  • {w}")
    else:
        lines.append("-" * 70)
        lines.append("Status: Environment configuration is optimized for JAX execution.")

    lines.append("=" * 70)
    out.write("\n".join(lines) + "\n")
    out.flush()
