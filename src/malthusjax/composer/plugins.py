"""Dynamic plugin and extension discovery for MalthusJAX.

Enables external packages (e.g., malthus-gp, malthus-neat, domain-specific extensions)
to register custom genomes, operators, engines, and CLI commands via standard Python
entry points without any hardcoded coupling in core MalthusJAX.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List

from malthusjax.core.logger import get_logger

logger = get_logger("composer.plugins")

_PLUGINS_LOADED = False


def load_plugins() -> None:
    """Discovers and imports all installed MalthusJAX extension plugins.

    Queries the 'malthusjax.plugins' entry-point group. Each entry point is
    loaded once, executing the extension's top-level module code and allowing it
    to call @register_genome, @register_operator, or @register_engine.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return

    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points

            eps = entry_points(group="malthusjax.plugins")
        else:
            import importlib_metadata as metadata  # type: ignore[no-redef]

            eps = metadata.entry_points().get("malthusjax.plugins", [])  # type: ignore[assignment]

        for ep in eps:
            try:
                ep.load()
                logger.debug("Successfully loaded MalthusJAX plugin: %s", ep.name)
            except Exception as e:
                logger.warning("Failed to load MalthusJAX plugin '%s': %s", ep.name, e)
    except Exception as e:
        logger.debug("Plugin discovery encountered an error: %s", e)

    _PLUGINS_LOADED = True


def discover_cli_commands() -> Dict[str, Callable[[List[str]], int]]:
    """Discovers CLI subcommand entry points exposed by installed extensions.

    Queries the 'malthusjax.commands' entry-point group. Each entry point must
    point to a callable accepting a list of string arguments and returning an exit code.

    Returns:
        Mapping of command name to entry point callable.
    """
    commands: Dict[str, Callable[[List[str]], int]] = {}

    try:
        if sys.version_info >= (3, 10):
            from importlib.metadata import entry_points

            eps = entry_points(group="malthusjax.commands")
        else:
            import importlib_metadata as metadata  # type: ignore[no-redef]

            eps = metadata.entry_points().get("malthusjax.commands", [])  # type: ignore[assignment]

        for ep in eps:
            try:
                func = ep.load()
                commands[ep.name] = func
                logger.debug("Discovered MalthusJAX CLI command: %s", ep.name)
            except Exception as e:
                logger.warning("Failed to load CLI command '%s': %s", ep.name, e)
    except Exception as e:
        logger.debug("CLI command discovery encountered an error: %s", e)

    return commands


__all__ = ["load_plugins", "discover_cli_commands"]
