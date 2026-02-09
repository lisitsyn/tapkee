# Tapkee

Python bindings for the Tapkee C++ dimensionality reduction library.

Tapkee provides efficient implementations of over 20 dimensionality reduction methods
with a focus on spectral techniques. It leverages Eigen for linear algebra and ARPACK
for large-scale eigenproblems.

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

- **Spectral:** Isomap, Landmark Isomap, Diffusion Maps, Laplacian Eigenmaps
- **Local:** LLE, Kernel LLE, Hessian LLE, LTSA, Linear LTSA, LPP, NPE
- **Linear:** PCA, Kernel PCA, Factor Analysis, Random Projection
- **Modern:** t-SNE, Barnes-Hut-SNE, Stochastic Proximity Embedding
- **Other:** MDS, Landmark MDS, Manifold Sculpting

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
