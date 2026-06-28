#!/usr/bin/env python3
"""Generate vcpkg.json with lapack-reference pinned to 3.11.0.

lapack-reference 3.12.x added sgedmd.f90/sgedmdq.f90 which use
USE ISO_FORTRAN_ENV - a module the Windows runner's flang can't find.
"""
import subprocess
import json

sha = subprocess.check_output(
    ['git', '-C', 'C:/vcpkg', 'rev-parse', 'HEAD']
).decode().strip()

config = {
    "dependencies": ["eigen3", "openblas", "arpack-ng", "fmt"],
    "overrides": [{"name": "lapack-reference", "version": "3.11.0"}],
    "builtin-baseline": sha,
}

with open('vcpkg.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"vcpkg.json written with baseline {sha}, lapack-reference pinned to 3.11.0")
