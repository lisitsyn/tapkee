# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tapkee is a C++ header-only template library for dimensionality reduction with focus on spectral methods. It integrates with Eigen3 for linear algebra and optionally ARPACK for eigendecomposition.

## Build Commands

```bash
# Standard build with examples
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON ..
make

# Build with tests
cmake -DBUILD_TESTS=ON ..
make

# Build Python bindings (nanobind)
cmake -DBUILD_NANOBIND=ON ..
make

# Build without GPL-licensed components (CoverTree)
cmake -DGPL_FREE=1 ..
```

**Key CMake options:**
- `BUILD_EXAMPLES` - Build example applications
- `BUILD_TESTS` - Build unit tests (requires GTest)
- `BUILD_NANOBIND` - Build Python extension
- `USE_GCOV` - Generate code coverage (GCC only)
- `GPL_FREE` - Exclude LGPLv3-licensed CoverTree

## Running Tests

```bash
# Run all tests
cd build && ctest -VV

# Run specific test suite
ctest -R "Methods" -VV

# Memory leak testing
./test/valgrind_run_all.sh
```

Tests are in `test/unit/` using Google Test framework.

## Architecture

### Core Components

```
include/tapkee/
├── chain_interface.hpp    # User-facing method chaining API
├── embed.hpp              # Core embedding orchestration
├── methods/               # 20+ dimensionality reduction algorithms
├── routines/              # Low-level algorithm implementations
├── callbacks/             # Data access abstraction layer
├── neighbors/             # Neighbor search (CoverTree, VP-tree)
├── utils/                 # ARPACK wrapper, matrix ops, logging
├── defines/               # Types, method enums, keywords
└── external/              # Barnes-Hut-SNE, CoverTree implementations
```

### Key Abstractions

1. **Chain Interface** (`chain_interface.hpp`): Primary API using method chaining pattern
2. **Callbacks**: Decouple algorithms from data storage - kernel, distance, features callbacks
3. **Parameters**: Keyword-based parameter passing via `stichwort` library using `with()`
4. **TapkeeOutput**: Return type containing embedding matrix and projection function

### API Pattern

```cpp
auto result = tapkee::with((method=Isomap, num_neighbors=15, target_dimension=2))
    .withDistance(distance_callback)
    .embedUsing(data);
```

### Implemented Methods

Spectral: Isomap, Landmark Isomap, Diffusion Maps, Laplacian Eigenmaps
Local: LLE, KLLE, Hessian LLE, LTSA, Linear LTSA, LPP, NPE
Linear: PCA, Kernel PCA, Factor Analysis, Random Projection
Modern: t-SNE, Barnes-Hut-SNE, Stochastic Proximity Embedding
Other: MDS, Landmark MDS, Manifold Sculpting

## Dependencies

- **Eigen3** (3.3+) - Required
- **ARPACK** - Optional but recommended for large-scale eigenproblems
- **fmt, cxxopts** - For CLI application
- **GTest** - For tests
- **nanobind** - For Python bindings

## Project Layout

- `include/tapkee/` - Header-only library (main code)
- `include/stichwort/` - External keyword parameter library
- `src/cli/` - Command-line application
- `src/python/` - Python nanobind bindings
- `examples/` - Usage examples (minimal, mnist, faces, etc.)
- `test/unit/` - Unit tests

## Customization

Preprocessor definitions (set before including Tapkee):
- `TAPKEE_CUSTOM_NUMTYPE` - Change internal type (default: double)
- `TAPKEE_USE_LGPL_COVERTREE` - Use LGPL-licensed CoverTree
- `TAPKEE_USE_FIBONACCI_HEAP` - Use Fibonacci heap instead of priority queue

## CLI Application

After building with `BUILD_EXAMPLES=ON`:
```bash
./bin/tapkee -h  # Show help with all options
./bin/tapkee --method isomap --neighbors 15 --target-dimension 2 -i input.dat -o output.dat
```
