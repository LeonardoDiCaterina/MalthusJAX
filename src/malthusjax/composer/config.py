from typing import Any, Dict, cast

try:
    import tomllib as _toml  # type: ignore
except ModuleNotFoundError:
    import tomli as _toml  # type: ignore

toml = _toml  # type: ignore[assignment]

def load_config(path: str, pipeline_name: str) -> Dict[str, Any]:
    """Very small TOML loader that returns the raw pipeline section.
    (Keep this lightweight—validation and pydantic will come later.)
    """
    try:
        with open(path, "rb") as f:
            cfg = toml.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e
    pipelines = cfg.get("pipelines", {})
    pipeline = pipelines.get(pipeline_name)
    if pipeline is None:
        raise KeyError(f"Pipeline '{pipeline_name}' not found in {path}")
    return cast(Dict[str, Any], pipeline)