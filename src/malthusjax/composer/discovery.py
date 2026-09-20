"""Plugin Auto-Discovery for MalthusJAX.

This module provides utilities to automatically discover and load MalthusJAX
extensions (engines, operators, genomes, etc.) without requiring the user
to manually import them.

Discovery happens in two places:
1. Python Entry Points: Scanning `importlib.metadata` for packages that
   have registered under the `malthusjax.plugins` group.
2. Local Directory: Scanning a local `plugins/` directory in the current
   working directory for `.py` files, which is useful for rapid prototyping.
"""

import importlib.metadata
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

# Track loaded plugins to avoid double-loading
_LOADED_PLUGINS: Set[str] = set()


def discover_plugins() -> None:
    """Discover and load all MalthusJAX plugins.

    This function is idempotent. It loads plugins via entry points and
    from a local `plugins/` directory.
    """
    _discover_entry_points()
    _discover_local_plugins()


def _discover_entry_points() -> None:
    """Load plugins registered via pyproject.toml entry points."""
    try:
        # Python 3.10+ syntax
        eps = importlib.metadata.entry_points(group="malthusjax.plugins")
    except TypeError:
        # Python 3.8/3.9 syntax
        eps = importlib.metadata.entry_points().get("malthusjax.plugins", [])  # type: ignore

    for ep in eps:
        if ep.name in _LOADED_PLUGINS:
            continue
        try:
            ep.load()
            _LOADED_PLUGINS.add(ep.name)
            logger.debug(f"Successfully loaded MalthusJAX plugin from entry point: {ep.name}")
        except Exception as e:
            logger.warning(f"Failed to load MalthusJAX plugin '{ep.name}': {e}")


def _discover_local_plugins() -> None:
    """Load .py files from a local 'plugins/' directory."""
    plugins_dir = Path(os.getcwd()) / "plugins"
    if not plugins_dir.is_dir():
        return

    # To allow plugins to import from each other, add plugins_dir to sys.path
    if str(plugins_dir) not in sys.path:
        sys.path.insert(0, str(plugins_dir))

    for py_file in plugins_dir.rglob("*.py"):
        # Ignore private files, __init__.py, and tests
        if (
            py_file.name.startswith("_")
            or py_file.name.startswith("test_")
            or "tests" in py_file.parts
        ):
            continue

        module_name = py_file.stem
        plugin_id = f"local:{module_name}"
        if plugin_id in _LOADED_PLUGINS:
            continue

        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                _LOADED_PLUGINS.add(plugin_id)
                logger.debug(f"Successfully loaded local plugin: {py_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load local MalthusJAX plugin '{py_file.name}': {e}")
