"""Zero-dependency, hierarchical logging subsystem for MalthusJAX.

Provides structured formatting, ANSI terminal colors, JSON output,
dynamic log-level control, and host-side callbacks for device telemetry (JAX JIT).
Adheres strictly to PEP 282 (library silence by default via NullHandler).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

_ROOT_LOGGER_NAME = "malthusjax"

# Standard LogRecord attributes to ignore when serializing extra kwargs to JSON
_STANDARD_RECORD_ATTRS: Set[str] = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
}


class MalthusColorFormatter(logging.Formatter):
    """ANSI color formatter for interactive terminal output.

    Colors:
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red
    """

    LEVEL_COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def __init__(
        self,
        show_timestamps: bool = False,
        use_color: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.show_timestamps = show_timestamps
        if use_color is None:
            # Auto-detect: only use color if stdout is a TTY and NO_COLOR is not set
            self.use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and "NO_COLOR" not in os.environ
        else:
            self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        msg = record.getMessage()

        # Simplify logger name: remove "malthusjax." prefix for cleaner output
        short_name = record.name
        if short_name.startswith(f"{_ROOT_LOGGER_NAME}."):
            short_name = short_name[len(_ROOT_LOGGER_NAME) + 1 :]
        elif short_name == _ROOT_LOGGER_NAME:
            short_name = "core"

        if self.use_color:
            color = self.LEVEL_COLORS.get(record.levelno, "")
            reset = self.RESET
            level_str = f"{color}{level_name:<8}{reset}"
        else:
            level_str = f"{level_name:<8}"

        prefix = f"[{level_str}] [{short_name}]"
        if self.show_timestamps:
            timestamp = datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"{timestamp} {prefix}"

        formatted = f"{prefix} {msg}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                formatted = f"{formatted}\n{record.exc_text}"

        return formatted


class MalthusJSONFormatter(logging.Formatter):
    """Newline-delimited JSON formatter for structured logging and pipeline analysis."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.datetime.fromtimestamp(
            record.created, tz=datetime.timezone.utc
        ).isoformat()
        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Include custom extra fields if attached to the record
        extras: Dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(val)
                    extras[key] = val
                except (TypeError, OverflowError):
                    extras[key] = str(val)

        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def _resolve_level(level: Union[str, int]) -> int:
    """Normalize string or integer log levels to logging integer constants."""
    if isinstance(level, int):
        return level
    name = str(level).strip().upper()
    level_val = getattr(logging, name, None)
    if not isinstance(level_val, int):
        raise ValueError(f"Invalid log level: {level!r}. Expected one of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    return level_val


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Obtain a logger under the hierarchical 'malthusjax' namespace.

    Parameters
    ----------
    name : Optional[str]
        Subsystem name, e.g. ``'engine'``, ``'operators'``, ``'benchmarking.runner'``.
        If omitted or None, returns the root ``'malthusjax'`` logger.

    Returns
    -------
    logging.Logger
        Hierarchical logger instance.
    """
    if not name:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def set_log_level(level: Union[str, int]) -> None:
    """Dynamically update the logging threshold for the entire MalthusJAX subsystem.

    Parameters
    ----------
    level : str or int
        New log level (e.g. ``'DEBUG'``, ``'INFO'``, ``logging.WARNING``).
    """
    lvl = _resolve_level(level)
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(lvl)
    for handler in root.handlers:
        handler.setLevel(lvl)


def configure_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_type: str = "color",
    show_timestamps: bool = False,
    force: bool = True,
) -> logging.Logger:
    """Configure console and optional file logging handlers for MalthusJAX.

    Parameters
    ----------
    level : str or int
        Initial log level (default: ``'INFO'``).
    log_file : str or Path, optional
        Destination file path for logging. If provided, creates parent directories.
    format_type : str
        ``'color'`` for ANSI-colored terminal output, or ``'json'`` for structured JSON.
    show_timestamps : bool
        Whether to prefix terminal logs with wall-clock timestamps (default: False).
    force : bool
        If True, clears previously configured MalthusJAX handlers to avoid duplicate lines.

    Returns
    -------
    logging.Logger
        The configured root ``'malthusjax'`` logger.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    lvl = _resolve_level(level)
    root.setLevel(lvl)

    # Clean up existing MalthusJAX-managed handlers if force=True
    if force:
        handlers_to_keep = []
        for h in root.handlers:
            if not getattr(h, "_is_malthusjax_managed", False):
                handlers_to_keep.append(h)
        root.handlers = handlers_to_keep

    # 1. Console stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler._is_malthusjax_managed = True  # type: ignore[attr-defined]
    stream_handler.setLevel(lvl)

    if format_type.lower() == "json":
        stream_handler.setFormatter(MalthusJSONFormatter())
    else:
        stream_handler.setFormatter(
            MalthusColorFormatter(show_timestamps=show_timestamps)
        )
    root.addHandler(stream_handler)

    # 2. File handler (optional)
    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler._is_malthusjax_managed = True  # type: ignore[attr-defined]
        file_handler.setLevel(lvl)

        if format_type.lower() == "json":
            file_handler.setFormatter(MalthusJSONFormatter())
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    return root


@dataclasses.dataclass(frozen=True)
class StepLoggingConfig:
    """Configuration for device-to-host JIT telemetry via ``jax.debug.callback``.

    Parameters
    ----------
    log_interval : Optional[int]
        Number of generations between telemetry dispatches.
        If ``None`` or ``<= 0``, step logging is disabled and eliminated at JIT trace time.
    log_nan_watchdog : bool
        If True, evaluates finiteness on GPU and dispatches an anomaly callback only when
        non-finite (NaN or Inf) values occur.
    logger_name : str
        Target logger channel for step telemetry messages.
    """

    log_interval: Optional[int] = None
    log_nan_watchdog: bool = True
    logger_name: str = "malthusjax.engine.step"

    def is_active(self) -> bool:
        """Return True if either step interval logging or NaN watchdog is enabled."""
        return (self.log_interval is not None and self.log_interval > 0) or self.log_nan_watchdog


def _host_log_step(
    gen: Any,
    best_fitness: Any,
    mean_fitness: Optional[Any] = None,
    logger_name: str = "malthusjax.engine.step",
) -> None:
    """Host callback invoked from JIT compiled loops via ``jax.debug.callback``."""
    logger = logging.getLogger(logger_name)
    try:
        g = int(gen)
    except Exception:
        g = gen

    try:
        bf = float(best_fitness)
        bf_str = f"{bf:10.4f}"
    except Exception:
        bf_str = str(best_fitness)

    if mean_fitness is not None:
        try:
            mf = float(mean_fitness)
            mf_str = f"{mf:10.4f}"
        except Exception:
            mf_str = str(mean_fitness)
        logger.info("[Gen %4s] Best Fitness: %s | Mean Fitness: %s", g, bf_str, mf_str)
    else:
        logger.info("[Gen %4s] Best Fitness: %s", g, bf_str)


def _host_log_nan_anomaly(
    gen: Any,
    val: Any,
    metric_name: str = "fitness",
    logger_name: str = "malthusjax.engine.anomaly",
) -> None:
    """Host callback invoked from JIT compiled loops when non-finite values are detected."""
    logger = logging.getLogger(logger_name)
    try:
        g = int(gen)
    except Exception:
        g = gen
    try:
        v = float(val)
        v_str = f"{v}"
    except Exception:
        v_str = str(val)
    logger.critical(
        "[CRITICAL] Non-finite %s detected at generation %s: %s",
        metric_name,
        g,
        v_str,
    )


def _init_from_env() -> None:
    """Inspect environment variables for automatic logging configuration."""
    env_level = os.environ.get("MALTHUSJAX_LOG_LEVEL")
    if env_level:
        try:
            configure_logging(level=env_level)
        except ValueError:
            pass


# Automatically inspect environment on module import
_init_from_env()
