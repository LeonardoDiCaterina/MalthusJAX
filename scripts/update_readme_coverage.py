import os
import re


def main():
    if not os.path.exists("coverage.md"):
        print("coverage.md not found!")
        return

    with open("coverage.md", "r") as f:
        coverage_data = f.read()

    with open("README.md", "r") as f:
        readme_content = f.read()

    # Find the block and replace it
    pattern = re.compile(r"(<!-- COVERAGE-START -->).*?(<!-- COVERAGE-END -->)", re.DOTALL)

    if not pattern.search(readme_content):
        print("Coverage block not found in README.md")
        return

    new_content = pattern.sub(f"\\1\n{coverage_data}\n\\2", readme_content)

    with open("README.md", "w") as f:
        f.write(new_content)

    print("README.md updated with latest coverage.")


if __name__ == "__main__":
    main()
