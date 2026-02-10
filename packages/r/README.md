# Tapkee

Efficient dimensionality reduction library with a focus on spectral methods.

Tapkee implements over 20 dimensionality reduction algorithms and provides
a simple R interface to all of them.

## Installation

### Prerequisites

The `fmt` C++ library headers must be installed:

```bash
# macOS
brew install fmt

# Ubuntu/Debian
sudo apt-get install libfmt-dev

# Fedora
sudo dnf install fmt-devel
```

Optionally, install ARPACK for faster eigendecomposition on large datasets:

```bash
# macOS
brew install arpack

# Ubuntu/Debian
sudo apt-get install libarpack2-dev
```

### Install from source

```r
# install.packages("remotes")
remotes::install_github("lisitsyn/tapkee", subdir = "packages/r")
```

## Usage

```r
library(tapkee)

# Generate a Swiss roll dataset
set.seed(42)
n <- 2000
t <- 1.5 * pi * (1 + 2 * runif(n))
x <- t * cos(t)
y <- 30 * runif(n)
z <- t * sin(t)
data <- rbind(x, y, z)  # 3 features, n samples

# Unroll with Locally Linear Embedding
embedding <- tapkee_embed(data, method = "lle", num_neighbors = 12L)

plot(embedding[, 1], embedding[, 2], col = rainbow(n)[rank(t)],
     pch = 20, cex = 0.5, main = "Swiss roll unrolled with LLE")
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
| Random Projection | `"random_projection"` | |
| MDS | `"mds"` | |
| Landmark MDS | `"l-mds"` | `landmark_ratio` |
| t-SNE | `"t-sne"` | `sne_perplexity`, `sne_theta` |
| Stochastic Proximity Embedding | `"spe"` | `spe_global_strategy`, `spe_num_updates`, `spe_tolerance` |
| Manifold Sculpting | `"manifold_sculpting"` | `num_neighbors`, `squishing_rate`, `max_iteration` |

All methods accept `target_dimension` (default: 2). Neighbor-based methods accept `num_neighbors` (default: 5).

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
