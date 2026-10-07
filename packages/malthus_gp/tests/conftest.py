import sys
from pathlib import Path

# Add packages/malthus_gp/src to sys.path
pkg_src = Path(__file__).parent.parent / "src"
if str(pkg_src) not in sys.path:
    sys.path.insert(0, str(pkg_src))
