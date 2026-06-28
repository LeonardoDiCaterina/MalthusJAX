from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Dash configuration requires TOML support. "
            "Please install 'tomli' (e.g., pip install tomli) if using Python < 3.11."
        )


def load_config(file_path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration file and recursively resolve includes.

    The `includes` key can be a list of relative paths. Included files are
    loaded first, and the current file overrides any identical keys.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "rb") as f:
        config = tomllib.load(f)

    includes = config.get("includes", [])
    if isinstance(includes, str):
        includes = [includes]

    merged_config: dict[str, Any] = {}

    # Load base configuration from includes
    for inc_path in includes:
        inc_resolved = (path.parent / inc_path).resolve()
        inc_config = load_config(inc_resolved)
        merged_config = _deep_merge(merged_config, inc_config)

    # Override with current file's config
    merged_config = _deep_merge(merged_config, config)

    # Remove includes from final merged output
    if "includes" in merged_config:
        del merged_config["includes"]

    return merged_config


def _deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """Deep merge dict2 into dict1. dict2 values override dict1."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
