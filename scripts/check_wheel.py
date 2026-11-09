"""Import a built wheel outside the checkout and block checkout-only plugins."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wheel", type=Path)
    p.add_argument("--module", default="malthusjax")
    p.add_argument("--dependency-wheel", type=Path)
    a = p.parse_args()
    wheel = str(a.wheel.resolve())
    code = """import sys, importlib.abc
sys.path.insert(0, sys.argv[1])
class NoPlugins(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == "plugins" or fullname.startswith("plugins."):
   raise ImportError("Checkout plugins must not be required")
sys.meta_path.insert(0, NoPlugins())
import malthusjax
assert ".whl/" in malthusjax.__file__, malthusjax.__file__
from malthusjax.core.fitness.composable.transforms.tensorneat import TensorNeatTransform
from malthusjax.composer.backends.map_elites import build_map_elites_engine
print("Isolated wheel imports passed:", malthusjax.__file__)
"""
    if a.module != "malthusjax":
        code = """import sys, importlib, importlib.abc
sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[3])
class NoOptionalImports(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split(".")[0] in {"sklearn", "matplotlib", "networkx", "graphviz"}:
   raise ImportError("Optional dependency imported eagerly: " + fullname)
sys.meta_path.insert(0, NoOptionalImports())
m = importlib.import_module(sys.argv[2])
assert ".whl/" in m.__file__, m.__file__
assert not any(name.startswith(m.__name__ + ".learn") for name in sys.modules)
print("Isolated wheel optional-import boundary passed:", m.__file__)
"""
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                code,
                wheel,
                a.module,
                str(a.dependency_wheel.resolve()) if a.dependency_wheel else "",
            ],
            cwd=directory,
            check=True,
        )


if __name__ == "__main__":
    main()
