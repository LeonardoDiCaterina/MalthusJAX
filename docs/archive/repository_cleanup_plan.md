# Repository Polish & Orphan Release Plan

To present a professional, clean, and accessible repository for the JAX Show and Tell, we will take a radical and highly effective approach: creating a pristine **orphan branch**. 

This will generate a brand new branch with **zero commit history**, completely dropping the weight of your year-long trial and error commits. It will look like a fresh, immaculate release.

## Proposed Changes

### 1. Create the Orphan Branch
- **Action**: I will create a new orphan branch called `release/show-and-tell` (or a name of your choice).
- **Result**: This branch will start with no commit history.

### 2. Selective Staging (The Cleanup)
Instead of meticulously deleting tracked files, we will simply *only stage what matters*.
- **Action**: We will stage the core project files (`src/`, `tests/`, `configs/`, `examples/showcase/`, `docs/`, `scripts/`, `pyproject.toml`, `Dockerfile`, etc.).
- **Action**: We will deliberately **exclude** and permanently remove from tracking:
  - `scratch/`
  - `scripts/_archive/`
  - `_archive_thesis_mds/`
- **Action**: We will preserve your custom IDE skills by moving `_archive_thesis_mds/SKILL.md` to the `.agents/skills/parity/SKILL.md` directory *before* we abandon the archive folder.

### 3. Open Source Community Standards
A public project seeking community feedback needs standard community files.
- **Action**: Create a clean `CONTRIBUTING.md` outlining how to set up the dev environment.
- **Action**: Create a standard `CODE_OF_CONDUCT.md`.

### 4. The Pristine Commit
- **Action**: We will make a single, clean "Initial commit for public release" containing only the polished codebase.
- **Action**: We will physically delete all the untracked `*.log` files from your local workspace to clean up your view.

## User Review Required

> [!CAUTION]  
> Creating an orphan branch is a great idea, but please confirm:
> 1. Are you okay with calling the new branch `release/show-and-tell`? 
> 2. You will still have your old history on the `main` branch if you ever need to reference it. Are you eventually planning to replace the remote `main` branch with this new history, or just present this specific branch?

## Verification Plan
1. Run `git status` and `tree -L 1` on the new orphan branch to ensure only the desired folders are tracked.
2. Confirm the successful creation of `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
