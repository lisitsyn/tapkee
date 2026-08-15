---
name: release
description: Use this skill before you cut a new Tapkee release. This skill helps you set the version number and check that the release is ready. This skill lists all files that must show the new version number before you push a `vX.Y.Z` tag.
---

# How to cut a Tapkee release

Tapkee does not have one single source for the version number. The version
number is in more than one file. You must update all these files together
before you create the release tag.

This skill gives you a checklist. This skill does not update the files for
you. Read the current version number in each file. Tell the user what you
must change for the target version. Do not edit a file before the user
confirms the target version.

## Step 1: Find the target version number

- Read the current version number in `packages/python/pyproject.toml`. Find
  the line `version = "..."`.
- Ask the user for the target version number if the user has not given it.
  Do not guess the version number. A wrong version number is difficult to
  correct. You must correct it in three files and in one git tag.

## Step 2: Set the new version number in these files

All these files must show the same version number, `X.Y.Z`:

- `packages/python/pyproject.toml`. Set `version = "X.Y.Z"`.
- `packages/r/DESCRIPTION`. Set `Version: X.Y.Z`.
- `include/tapkee/defines.hpp`. Set `TAPKEE_MAJOR_VERSION` to `Y`. Set
  `TAPKEE_MINOR_VERSION` to `Z`. Do not change `TAPKEE_WORLD_VERSION`. Change
  this macro only for a breaking change or a new epoch. This macro has
  stayed at `1` since the start of the project.

Search the repository for the old version number. This search can find
other files with the old version number, for example README.md:

```bash
grep -rn "<old-version>" --include="*.toml" --include="DESCRIPTION" --include="*.hpp" --include="*.md" . \
  | grep -vE "build/|\.venv|venv/|dist/|tapkee\.Rcheck|__pycache__"
```

## Step 3: Check the repository before you create the tag

- Run `git status`. Check that the change set has only the version update.
  Do not commit build output, for example `tapkee.Rcheck/`,
  `tapkee_*.tar.gz`, or `examples/__pycache__/`. This build output is old
  local data. This build output is not part of the release.
- Check README.md for old links. An old link can point to an old file path.
  An old link can point to the `master` branch. The default branch is now
  `main`. Use this command: `grep -n "master\b" README.md`
- This repository does not have a `CHANGELOG.md` file. Do not create this
  file unless the user asks for it.

## Step 4: Create the tag and start the release

The file `.github/workflows/release.yml` starts the release process. This
process starts when you push a tag with the format `v*`. The process builds
the CLI binaries. The process builds the sdist package. The process builds
the wheels. The process publishes the package to PyPI. The process creates
the GitHub release.

Do this after the version-update commit reaches the `main` branch:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Ask the user before you push the tag. This action starts a real publish
process. This action is not reversible. The action publishes the package to
PyPI. The action creates a GitHub release.
