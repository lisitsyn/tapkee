# Tapkee

Efficient dimensionality reduction library with a focus on spectral methods.

Tapkee implements over 20 dimensionality reduction algorithms and provides
a simple Python interface to all of them.

## Installation

```bash
pip install tapkee
```

## Usage

```python
import numpy as np
import tapkee

data = np.random.randn(3, 100)  # 3 features, 100 samples
embedding = tapkee.embed(data, method="isomap", num_neighbors=15)
```

## Supported methods

| Method | `method=` | Key parameters |
|--------|-----------|----------------|
| Isomap | `"isomap"` | `num_neighbors` |
| Landmark Isomap | `"l-isomap"` | `num_neighbors`, `landmark_ratio` |
| Diffusion Maps | `"dm"` | `gaussian_kernel_width`, `diffusion_map_timesteps` |
| Laplacian Eigenmaps | `"la"` | `num_neighbors`, `gaussian_kernel_width` |
| Locally Linear Embedding | `"lle"` | `num_neighbors` |
| Hessian LLE | `"hlle"` | `num_neighbors` |
| Local Tangent Space Alignment | `"ltsa"` | `num_neighbors` |
| Linear LTSA | `"lltsa"` | `num_neighbors` |
| Locality Preserving Projections | `"lpp"` | `num_neighbors`, `gaussian_kernel_width` |
| Neighborhood Preserving Embedding | `"npe"` | `num_neighbors` |
| PCA | `"pca"` | |
| Kernel PCA | `"kpca"` | `gaussian_kernel_width` |
| Factor Analysis | `"fa"` | `max_iteration`, `fa_epsilon` |
| Random Projection | `"ra"` | |
| MDS | `"mds"` | |
| Landmark MDS | `"l-mds"` | `landmark_ratio` |
| t-SNE | `"t-sne"` | `sne_perplexity`, `sne_theta` |
| Stochastic Proximity Embedding | `"spe"` | `spe_global_strategy`, `spe_num_updates`, `spe_tolerance` |
| Manifold Sculpting | `"manifold_sculpting"` | `num_neighbors`, `squishing_rate`, `max_iteration` |

All methods accept `target_dimension` (default: 2). Neighbor-based methods accept `num_neighbors` (default: 5).

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_dimension` | int | Output dimensionality |
| `num_neighbors` | int | Number of neighbors for local methods |
| `gaussian_kernel_width` | float | Width of the Gaussian kernel |
| `landmark_ratio` | float | Ratio of landmark points (0 to 1) |
| `max_iteration` | int | Maximum iterations for iterative methods |
| `diffusion_map_timesteps` | int | Number of timesteps for Diffusion Maps |
| `sne_perplexity` | float | Perplexity for t-SNE |
| `sne_theta` | float | Angle for Barnes-Hut approximation in t-SNE |
| `squishing_rate` | float | Rate for Manifold Sculpting |
| `spe_global_strategy` | bool | Use global strategy for SPE |
| `spe_num_updates` | int | Number of SPE updates |
| `spe_tolerance` | float | SPE convergence tolerance |
| `nullspace_shift` | float | Regularizer for eigenproblems |
| `klle_shift` | float | KLLE regularizer |
| `fa_epsilon` | float | Factor Analysis convergence epsilon |
| `check_connectivity` | bool | Check graph connectivity |

## Links

- [Documentation](https://tapkee.lisitsyn.me)
- [Source code](https://github.com/lisitsyn/tapkee)
- [Issue tracker](https://github.com/lisitsyn/tapkee/issues)

## Citation

If you use Tapkee in a publication, please cite:

> Sergey Lisitsyn, Christian Widmer, Fernando J. Iglesias Garcia.
> Tapkee: An Efficient Dimension Reduction Library.
> Journal of Machine Learning Research, 14: 2355-2359, 2013.

## License

BSD 3-Clause
