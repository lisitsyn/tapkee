Tapkee is a C++ template library for dimensionality reduction with some bias on
spectral methods. The Tapkee origins from the code developed during
[GSoC 2011](http://www.google-melange.com/gsoc/homepage/google/gsoc2011) as
the part of the [Shogun machine learning toolbox](https://github.com/shogun-toolbox/shogun).
The project aim is to provide efficient and flexible standalone library for
dimensionality reduction which can be easily integrated to existing codebases.
Tapkee leverages capabilities of effective [Eigen linear algebra library](http://eigen.tuxfamily.org) and
optionally makes use of the [ARPACK eigensolver](http://www.caam.rice.edu/software/ARPACK/).
The library uses CoverTree and VP-tree data structures to compute nearest neighbors. To achieve
greater flexibility we provide a callback interface which decouples dimension reduction algorithms from
the data representation and storage schemes.

The library is distributed under permissive
[BSD 3-clause license](https://github.com/lisitsyn/tapkee/blob/master/include/LICENSE)
(except a few rather optional parts that are distributed under other
open sources licenses, see Licensing section of this document). If you use
this software in any publication we would be happy if you cite the following paper:

	Sergey Lisitsyn and Christian Widmer and Fernando J. Iglesias Garcia. Tapkee: An Efficient Dimension Reduction Library. Journal of Machine Learning Research, 14: 2355-2359, 2013.

Python
------

Install with pip:

	pip install tapkee

Usage:

```python
import numpy as np
import tapkee

# Generate a Swiss roll dataset (3 x 1000)
t = np.sort(3 * np.pi / 2 * (1 + 2 * np.random.rand(1000)))
data = np.array([t * np.cos(t), 10 * np.random.rand(1000), t * np.sin(t)])

# Embed with LLE (default method)
embedding = tapkee.embed(data, method='lle', num_neighbors=15)

# Embed with Isomap using vptree neighbor search
embedding = tapkee.embed(data, method='isomap',
                         neighbors_method='vptree',
                         num_neighbors=15, target_dimension=2)

# Embed with PCA using randomized eigendecomposition
embedding = tapkee.embed(data, method='pca',
                         eigen_method='randomized',
                         target_dimension=3)
```

The `embed()` function accepts:
- `method`: reduction algorithm (`'lle'`, `'isomap'`, `'pca'`, `'t-sne'`, `'mds'`, `'dm'`, `'kpca'`, `'la'`, `'lpp'`, `'npe'`, `'ltsa'`, `'lltsa'`, `'hlle'`, `'l-mds'`, `'l-isomap'`, `'spe'`, `'fa'`, `'random_projection'`, `'manifold_sculpting'`)
- `neighbors_method`: neighbor search strategy (`'brute'`, `'vptree'`, `'covertree'`)
- `eigen_method`: eigendecomposition method (`'dense'`, `'randomized'`, `'arpack'`)
- `num_neighbors`, `target_dimension`, `gaussian_kernel_width`, `sne_perplexity`, and other algorithm-specific parameters

R
-

The R package provides a single `tapkee_embed()` function:

```r
library(tapkee)

data <- matrix(rnorm(3000), nrow = 3, ncol = 1000)

# LLE with default settings
result <- tapkee_embed(data)

# Isomap with vptree neighbor search
result <- tapkee_embed(data, method = "isomap",
                       neighbors_method = "vptree",
                       num_neighbors = 15L, target_dimension = 2L)

# PCA with dense eigendecomposition
result <- tapkee_embed(data, method = "pca",
                       eigen_method = "dense",
                       target_dimension = 3L)
```

C++ API
-------

We provide an interface based on the method chaining technique. The chain starts with the call
of the `with(const ParametersSet&)` method, which is used to provide parameters like the method
to use and its settings. The provided argument is formed with the following syntax:

	(keyword1=value1, keyword2=value2)

Such syntax is possible due to comma operator overloading which groups all assigned keywords
in the comma separated list.

Keywords are defined in the `tapkee` namespace. Currently, the following keywords
are defined: `method`, `eigen_method`, `neighbors_method`, `num_neighbors`, `target_dimension`,
`diffusion_map_timesteps`, `gaussian_kernel_width`, `max_iteration`, `spe_global_strategy`,
`spe_num_updates`, `spe_tolerance`, `landmark_ratio`, `nullspace_shift`, `klle_shift`,
`check_connectivity`, `fa_epsilon`, `progress_function`, `cancel_function`, `sne_perplexity`,
`sne_theta`. See the documentation for their detailed meaning.

As an example of parameters setting, if you want to use the Isomap
algorithm with the number of neighbors set to 15:

	tapkee::with((method=Isomap,num_neighbors=15))

Please note that the inner parentheses are necessary as it uses the
comma operator which appears to be ambiguous in this case.

Next, you may either embed the provided matrix with:

	tapkee::with((method=Isomap,num_neighbors=15)).embedUsing(matrix);

Or provide callbacks (kernel, distance and features) using any combination
of the `withKernel(KernelCallback)`, `withDistance(DistanceCallback)` and
`withFeatures(FeaturesCallback)` member functions:

	tapkee::with((method=Isomap,num_neighbors=15))
	       .withKernel(kernel_callback)
	       .withDistance(distance_callback)
	       .withFeatures(features_callback)

Once callbacks are initialized you may either embed data using an
STL-compatible sequence of indices or objects (that supports the
`begin()` and `end()` methods to obtain the corresponding iterators)
with the `embedUsing(Sequence)` member function
or embed the data using a sequence range with the
`embedRange(RandomAccessIterator, RandomAccessIterator)`
member function.

As a summary - a few examples:

	TapkeeOutput output = with((method=Isomap,num_neighbors=15))
	    .embedUsing(matrix);

	TapkeeOutput output = with((method=Isomap,num_neighbors=15))
	    .withDistance(distance_callback)
	    .embedUsing(indices);

	TapkeeOutput output = with((method=Isomap,num_neighbors=15))
	    .withDistance(distance_callback)
	    .embedRange(indices.begin(),indices.end());

Minimal example
---------------

A minimal working example of a program that uses the library is:

	#include <tapkee/tapkee.hpp>
	#include <tapkee/callbacks/dummy_callbacks.hpp>

	using namespace std;
	using namespace tapkee;

	struct MyDistanceCallback
	{
		ScalarType distance(IndexType l, IndexType r) { return abs(l-r); }
	};

	int main(int argc, const char** argv)
	{
		const int N = 100;
		vector<IndexType> indices(N);
		for (int i=0; i<N; i++) indices[i] = i;

		MyDistanceCallback d;

		TapkeeOutput output = tapkee::with((method=MultidimensionalScaling,target_dimension=1))
		   .withDistance(d)
		   .embedUsing(indices);

		cout << output.embedding.transpose() << endl;
		return 0;
	}

This example require Tapkee to be in the include path. With Linux compilers
you may do that with the `-I/path/to/tapkee/headers/folder` key.

Integration
-----------

There are a few issues related to including the Tapkee library to your code. First, if your library
already includes Eigen (and only if) - you might need to let Tapkee
know about that with the following define:

`#define TAPKEE_EIGEN_INCLUDE_FILE <path/to/your/eigen/include/file.h>`

Please note that if you don't include Eigen in your project there is no need to define that variable -
in this case Eigen will be included by Tapkee. This issue comes from the need of including the Eigen library
only once when using some specific parameters (like debug and extensions).

If you are able to use less restrictive licenses (such as LGPLv3) you may define
the following variable:

- `TAPKEE_USE_LGPL_COVERTREE` to use Covertree code by John Langford.

When compiling your software that includes Tapkee be sure Eigen headers are in include path and your code
is linked against ARPACK library (-larpack key for g++ and clang++).

When working with installed headers you may check which version of the library
do you have with checking the values of `TAPKEE_WORLD_VERSION`, `TAPKEE_MAJOR_VERSION`
and `TAPKEE_MINOR_VERSION` defines.

We welcome any integration so please contact authors if you have got any questions. If you have
successfully used the library please also let authors know about that - mentions of any
applications are very appreciated.

Customization
-------------

Tapkee is designed to be highly customizable with preprocessor definitions.

If you want to use float as internal numeric type (default is double) you may do
that with definition of `#define TAPKEE_CUSTOM_NUMTYPE float`
before including [defines header](https://github.com/lisitsyn/tapkee/blob/master/include/tapkee_defines.hpp).

If you use some non-standard STL-compatible realization of vector, map and pair you may redefine them
with `TAPKEE_INTERNAL_VECTOR`, `TAPKEE_INTERNAL_PAIR`, `TAPKEE_INTERNAL_MAP`
(they are set to std::vector, std::pair and std::map by default otherwise).

You may define `TAPKEE_USE_FIBONACCI_HEAP` or `TAPKEE_USE_PRIORITY_QUEUE` to select which
data structure should be used in the shortest paths computing algorithm. By default
a priority queue is used.

Other properties can be loaded from some provided header file using `#define TAPKEE_CUSTOM_PROPERTIES`. Currently
such file should define only one variable - `COVERTREE_BASE` which defines the base of the CoverTree (default is 1.3).

Building from source
--------------------

The library requires Eigen 3.4+, fmt, and a C++20 compiler. ARPACK is optional but
recommended for large-scale eigenproblems.

On Debian/Ubuntu:

	sudo apt-get install libeigen3-dev libarpack2-dev libfmt-dev

On macOS with Homebrew:

	brew install eigen arpack fmt

Build the command line application and examples:

	mkdir build && cd build
	cmake -DBUILD_EXAMPLES=ON ..
	make

Build with tests:

	cmake -DBUILD_TESTS=ON ..
	make
	ctest -VV

Build Python bindings from source:

	cmake -DBUILD_NANOBIND=ON ..
	make

Other CMake options:
- `-DGPL_FREE=1` - build without LGPLv3-licensed CoverTree
- `-DUSE_GCOV=1` - enable code coverage (GCC only)

Command line application
-----------

Tapkee comes with a sample application which can be used to construct
low-dimensional representations of dense feature matrices. For more information on
its usage please run:

`./bin/tapkee -h`

The application takes plain ASCII file containing dense matrix (each vector is a column and each
line contains values of some feature). The output of the application is stored into the provided
file in the same format (each line is feature).

Directory contents
------------------

The repository of Tapkee contains the following directories:

- `include/tapkee/` - the header-only C++ library itself.
- `include/stichwort/` - keyword parameter library used by Tapkee.
- `src/cli/` - command-line application.
- `src/python/` - Python nanobind bindings.
- `src/R/` - R (Rcpp) binding source.
- `packages/python/` - Python package configuration (pyproject.toml).
- `packages/r/` - R package (DESCRIPTION, R wrappers, tests).
- `test/unit/` - C++ unit tests (Google Test).
- `test/` - Python extension tests.
- `examples/` - usage examples (minimal, mnist, faces, etc.).
- `data/` - a git submodule with example data.
  Use `git submodule update --init` to initialize.

Need help?
----------

If you need any help or advice don't hesitate to send [an email](mailto://lisitsyn@hey.com) or
fire [an issue at github](https://github.com/lisitsyn/tapkee/issues/new).

Supported platforms
-------------------

Tapkee is tested on Linux (GCC, Clang), macOS (Clang), and Windows (MSVC)
via GitHub Actions CI. The library requires C++20.

Supported dimension reduction methods
-------------------------------------

Tapkee provides implementations of the following dimension reduction methods:

* Locally Linear Embedding and Kernel Locally Linear Embedding (LLE/KLLE)
* Neighborhood Preserving Embedding (NPE)
* Local Tangent Space Alignment (LTSA)
* Linear Local Tangent Space Alignment (LLTSA)
* Hessian Locally Linear Embedding (HLLE)
* Laplacian eigenmaps
* Locality Preserving Projections
* Diffusion map
* Isomap and landmark Isomap
* Multidimensional scaling and landmark Multidimensional scaling (MDS/lMDS)
* Stochastic Proximity Embedding (SPE)
* Principal Component Analysis (PCA)
* Kernel Principal Component Analysis (PCA)
* Random projection
* Factor analysis
* t-SNE
* Barnes-Hut-SNE
* Manifold Sculpting

Licensing
---------

The library is distributed under the [BSD 3-clause](https://github.com/lisitsyn/tapkee/blob/master/LICENSE) license.

Exceptions are:

- [Barnes-Hut-SNE code](https://github.com/lisitsyn/tapkee/blob/master/include/external/barnes_hut_sne/) by Laurens van der Maaten which
  is distributed under the BSD 4-clause license.

- [Covertree code](https://github.com/lisitsyn/tapkee/blob/master/include/neighbors/covertree.hpp) by John Langford and Dinoj Surendran
  which is distributed under the [LGPLv3 license](https://github.com/lisitsyn/tapkee/blob/master/LGPL-LICENSE).
