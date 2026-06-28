#!/usr/bin/env python3
"""Patch vcpkg's lapack-reference port to remove Fortran files that fail with VS 2026's flang.

lapack-reference 3.12.x added sgedmd.f90 and sgedmdq.f90 which use
USE ISO_FORTRAN_ENV — a Fortran 2003 module the bundled flang cannot find.
The 3.12.x portfile has correct C compiler setup for cmake 4.x / VS 2026,
so we copy it and patch out only the two problematic source files.
"""
import shutil
import pathlib

src_port = pathlib.Path('C:/vcpkg/ports/lapack-reference')
overlay_dir = pathlib.Path('lapack-overlay/lapack-reference')
if overlay_dir.exists():
    shutil.rmtree(overlay_dir)
overlay_dir.parent.mkdir(exist_ok=True)
shutil.copytree(src_port, overlay_dir)

portfile = overlay_dir / 'portfile.cmake'
content = portfile.read_text(encoding='utf-8')

# Inject cmake commands before vcpkg_cmake_configure to:
# 1. Delete the .f90 files so the compiler never sees them
# 2. Patch SRC/CMakeLists.txt so cmake doesn't list them as sources
removal = r"""
# Patch: sgedmd.f90/sgedmdq.f90 (added in LAPACK 3.12.x) use USE ISO_FORTRAN_ENV
# which VS 2026 flang cannot resolve. Remove files and strip from CMakeLists.txt.
file(REMOVE "${SOURCE_PATH}/SRC/sgedmd.f90" "${SOURCE_PATH}/SRC/sgedmdq.f90")
file(READ "${SOURCE_PATH}/SRC/CMakeLists.txt" _lapack_src_cmake)
string(REPLACE "\r\n" "\n" _lapack_src_cmake "${_lapack_src_cmake}")
string(REGEX REPLACE "[ \t]*sgedmd\\.f90[ \t]*\n" "" _lapack_src_cmake "${_lapack_src_cmake}")
string(REGEX REPLACE "[ \t]*sgedmdq\\.f90[ \t]*\n" "" _lapack_src_cmake "${_lapack_src_cmake}")
file(WRITE "${SOURCE_PATH}/SRC/CMakeLists.txt" "${_lapack_src_cmake}")

"""

patched = content.replace('vcpkg_cmake_configure(', removal + 'vcpkg_cmake_configure(', 1)
if patched == content:
    raise RuntimeError("Could not find 'vcpkg_cmake_configure(' in portfile.cmake")

portfile.write_text(patched, encoding='utf-8')
print(f"Created overlay at {overlay_dir}: sgedmd files removed and stripped from CMakeLists.txt")

# Clean up stale vcpkg.json from previous approach
stale = pathlib.Path('vcpkg.json')
if stale.exists():
    stale.unlink()
