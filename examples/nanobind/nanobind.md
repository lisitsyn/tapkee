# Nanobind Python Bindings Example

This example demonstrates how to use Tapkee's Python bindings (built with nanobind) for dimensionality reduction.

## Prerequisites

Build Tapkee with Python bindings enabled:

```bash
mkdir build && cd build
cmake -DBUILD_NANOBIND=ON ..
make
```

This creates the `tapkee` module in `lib/tapkee.cpython-*.so`.

## Usage

Run from the repository root:

```bash
python examples/nanobind/nanobind_example.py
```

Or with different demos:

```bash
python examples/nanobind/nanobind_example.py --demo compare
python examples/nanobind/nanobind_example.py --demo datasets
python examples/nanobind/nanobind_example.py --demo all
```

## Available Methods

The bindings support all Tapkee methods including:

- **Spectral methods**: `isomap`, `l-isomap`, `dm` (Diffusion Map), `la` (Laplacian Eigenmaps)
- **Local methods**: `lle`, `hlle`, `ltsa`, `lltsa`, `lpp`, `npe`
- **Linear methods**: `pca`, `kpca`, `fa` (Factor Analysis), `ra` (Random Projection)
- **Modern methods**: `t-sne`, `spe`, `manifold_sculpting`
- **MDS variants**: `mds`, `l-mds`

## API

### Simple API (recommended)

```python
import sys
sys.path.insert(0, 'lib')
import tapkee

# data should be DxN matrix (features as rows, samples as columns)
embedding = tapkee.embed(data, method='lle', num_neighbors=15, target_dimension=2)
```

All parameters are optional with sensible defaults:

```python
# Minimal usage - uses LLE with default settings
embedding = tapkee.embed(data)

# Specify method and key parameters
embedding = tapkee.embed(data, method='isomap', num_neighbors=10)

# Use method-specific parameters
embedding = tapkee.embed(data, method='t-sne', sne_perplexity=40)
embedding = tapkee.embed(data, method='l-mds', landmark_ratio=0.3)
```

### Low-level API

For full control, use the parameter-based API:

```python
import tapkee

params = tapkee.ParametersSet()
params.add(tapkee.Parameter.create('dimension reduction method',
                                   tapkee.parse_reduction_method('lle')))
params.add(tapkee.Parameter.create('number of neighbors', 15))
embedding = tapkee.withParameters(params).embedUsing(data).embedding
```

## Parameters

All parameters for `tapkee.embed()`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | 'lle' | Reduction method name |
| `num_neighbors` | int | 5 | k for local methods |
| `target_dimension` | int | 2 | Output dimensionality |
| `gaussian_kernel_width` | float | 1.0 | For Laplacian Eigenmaps, DM, LPP |
| `landmark_ratio` | float | 0.5 | For l-isomap, l-mds (0-1) |
| `max_iteration` | int | 100 | For iterative methods |
| `diffusion_map_timesteps` | int | 3 | For Diffusion Map |
| `sne_perplexity` | float | 30.0 | For t-SNE |
| `sne_theta` | float | 0.5 | For Barnes-Hut t-SNE |
| `squishing_rate` | float | 0.99 | For Manifold Sculpting |
| `spe_global_strategy` | bool | True | SPE global vs local |
| `spe_num_updates` | int | 100 | SPE updates |
| `spe_tolerance` | float | 1e-9 | SPE tolerance |
| `nullspace_shift` | float | 1e-9 | Eigenproblem regularizer |
| `klle_shift` | float | 1e-3 | KLLE regularizer |
| `fa_epsilon` | float | 1e-9 | Factor Analysis epsilon |
| `check_connectivity` | bool | True | Check graph connectivity |
