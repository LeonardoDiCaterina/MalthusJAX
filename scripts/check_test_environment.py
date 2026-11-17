"""Fail early when a test environment mixes incompatible JAX GPU packages."""

import sys
from importlib.metadata import PackageNotFoundError, version


def main() -> int:
    jaxlib = version("jaxlib")
    mismatches = []
    for package in ("jax-cuda12-plugin", "jax-cuda13-plugin", "jax-cuda12-pjrt", "jax-cuda13-pjrt"):
        try:
            installed = version(package)
        except PackageNotFoundError:
            continue
        if installed.split("+")[0] != jaxlib.split("+")[0]:
            mismatches.append(f"{package}={installed}")
    if mismatches:
        print(
            f"Incompatible GPU test environment: jaxlib={jaxlib}, {', '.join(mismatches)}.\n"
            "Install the package and its CUDA extra together, or align the existing stack with\n"
            f"  {sys.executable} -m pip install 'jax[cuda13]=={jaxlib}'\n"
            "Use cuda12 in that command for a CUDA 12 environment.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
