---
name: release
description: Use when preparing to cut a new Tapkee release (bumping the version, checking a release is ready to tag). Lists every place the version number and related metadata must be updated, in sync, before pushing a `vX.Y.Z` tag.
---

# Cutting a Tapkee release

Tapkee has no single source of truth for its version — it is duplicated across a
few files that must all be bumped together before tagging. This skill is a
checklist, not an automation: read the current state of each file, report
what needs to change for the target version, and only edit after confirming
the target version with the user.

## 1. Determine the target version

- Read the current version from `packages/python/pyproject.toml` (`version = "..."`).
- Ask the user for the target version if it isn't already clear from the
  conversation (patch/minor/major bump). Don't guess silently — a wrong bump
  here is annoying to unwind across three files plus a git tag.

## 2. Files that must be updated to the new version

All of these must match `X.Y.Z` of the target release:

- `packages/python/pyproject.toml` — `version = "X.Y.Z"`
- `packages/r/DESCRIPTION` — `Version: X.Y.Z`
- `include/tapkee/defines.hpp` — `TAPKEE_MAJOR_VERSION` and
  `TAPKEE_MINOR_VERSION` (map to `Y` and `Z`). Leave `TAPKEE_WORLD_VERSION`
  alone unless this is a deliberate breaking/epoch bump — it has stayed `1`
  since the project's origin.

Grep for the current version string across the repo to catch anything missed
(README badges/examples, `packages/r/DESCRIPTION`, changelog, etc.):

```bash
grep -rn "<old-version>" --include="*.toml" --include="DESCRIPTION" --include="*.hpp" --include="*.md" . \
  | grep -vE "build/|\.venv|venv/|dist/|tapkee\.Rcheck|__pycache__"
```

## 3. Sanity checks before tagging

- `git status` — make sure the working tree only contains the intended
  version-bump changes. Untracked build cruft (`tapkee.Rcheck/`,
  `tapkee_*.tar.gz`, `examples/__pycache__/`) should NOT be committed —
  it's stale local build output, not release material.
- Confirm README.md doesn't have stale links pointing at old file paths or
  the `master` branch (the default branch is `main`) — check with:
  `grep -n "master\b" README.md`
- No `CHANGELOG.md` is currently tracked in this repo — don't invent one
  unless the user asks for it.

## 4. Tag and trigger the release

`.github/workflows/release.yml` triggers on push of a `v*` tag and builds
CLI binaries, the sdist, wheels, and publishes to PyPI + creates the GitHub
release. After the version-bump commit is merged to `main`:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Confirm with the user before pushing the tag — it's a one-way trigger for a
real publish (PyPI + GitHub release), not a reversible local action.
