"""Check local links and the Sphinx navigation used by the release entry points."""

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/source"
PUBLIC_PAGES = (
    "README.md",
    "CONTRIBUTING.md",
    "examples/README.md",
    "docs/source/quickstart.md",
    "docs/source/research_workflows.md",
    "docs/source/toml_guide.md",
    "docs/source/extension_cli_guide.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^)\s]+)\)")


def main() -> int:
    missing = []
    pages = sys.argv[1:] or PUBLIC_PAGES
    for name in pages:
        page = ROOT / name
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
            for target in MARKDOWN_LINK.findall(line):
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                if not (page.parent / unquote(url.path)).exists():
                    missing.append(f"{name}:{number}: missing {target}")

    index = SOURCE / "index.rst"
    for number, line in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.strip()
        if not line.startswith("   ") or not entry or entry.startswith(":"):
            continue
        if not any((SOURCE / f"{entry}{ext}").is_file() for ext in (".rst", ".md")):
            missing.append(f"docs/source/index.rst:{number}: missing toctree entry {entry}")

    if missing:
        print("\n".join(missing), file=sys.stderr)
        return 1
    print("Public documentation links and toctree targets resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
