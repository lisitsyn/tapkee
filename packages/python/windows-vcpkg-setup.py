#!/usr/bin/env python3
"""Patch vcpkg's lapack-reference port to remove Fortran files that fail with VS 2026's flang.

lapack-reference 3.12.x added sgedmd.f90 and sgedmdq.f90 which use
USE ISO_FORTRAN_ENV — a Fortran 2003 module the bundled flang cannot find.
The 3.12.x portfile has correct C compiler setup for cmake 4.x / VS 2026,
so we copy it and remove only the two problematic source files.
"""
import shutil
import pathlib
import os

src_port = pathlib.Path('C:/vcpkg/ports/lapack-reference')
overlay_dir = pathlib.Path('lapack-overlay/lapack-reference')
if overlay_dir.exists():
    shutil.rmtree(overlay_dir)
overlay_dir.parent.mkdir(exist_ok=True)
shutil.copytree(src_port, overlay_dir)

portfile = overlay_dir / 'portfile.cmake'
content = portfile.read_text(encoding='utf-8')

removal = (
    '\n# Patch: sgedmd.f90 and sgedmdq.f90 (added in 3.12.x) use USE ISO_FORTRAN_ENV\n'
    '# which VS 2026 flang cannot resolve. Remove them before building.\n'
    'file(REMOVE "${SOURCE_PATH}/SRC/sgedmd.f90" "${SOURCE_PATH}/SRC/sgedmdq.f90")\n\n'
)

patched = content.replace('vcpkg_cmake_configure(', removal + 'vcpkg_cmake_configure(', 1)
if patched == content:
    raise RuntimeError("Could not find 'vcpkg_cmake_configure(' in portfile.cmake - overlay patch failed")

portfile.write_text(patched, encoding='utf-8')
print(f"Created overlay at {overlay_dir}: sgedmd Fortran files will be removed before build")

# Remove any stale vcpkg.json from a previous approach (classic mode needs none)
stale = pathlib.Path('vcpkg.json')
if stale.exists():
    stale.unlink()
    print("Removed stale vcpkg.json")
