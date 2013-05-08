Tapkee is a C++ template library for dimensionality reduction with some bias on 
spectral methods. The Tapkee origins from the code developed during 
[GSoC 2011](http://www.google-melange.com/gsoc/homepage/google/gsoc2011) as 
the part of the [Shogun machine learning toolbox](https://github.com/shogun-toolbox/shogun). 
The project aim is to provide efficient and flexible standalone library for 
dimensionality reduction which can be easily integrated to existing codebases.
Tapkee leverages capabilities of effective [Eigen3 linear algebra library](http://eigen.tuxfamily.org) and 
optionally makes use of the [ARPACK eigensolver](http://www.caam.rice.edu/software/ARPACK/). To achieve 
greater flexibility we provide a callback interface which decouples dimension reduction algorithms from
the data representation and storage schemes.

The library is distributed under permissive 
[BSD 3-clause license](https://github.com/lisitsyn/tapkee/tree/master/include/LICENSE) 
(except a few rather optional parts that are distributed under other 
open sources licenses, see Licensing section of this document).

To get started with dimension reduction you may try the 
[borsch script](https://github.com/lisitsyn/tapkee/tree/master/examples/borsch/borsch.py)
that embeds common datasets (swissroll, helix, scurve) using
the Tapkee library and outputs it with the help of 
Matplotlib library. To use the script build the 
sample application (see the Application section for more details) 
and call borsch with the following command:

	./examples/borsch [swissroll|helix|scurve|...] [lle|isomap|...]

You may also try out an minimal example using `make minimal` 
and the RNA example using `make rna`. There are a few graphical
examples. To run MNIST digits embedding example use `make mnist`,
to run promoters embedding example use `make promoters` and 
to run embedding for two images datasets use `make cbcl` and
`make faces`. All graphical examples require Matplotlib which can 
be usually installed with a package manager. The promoters example 
also has non-trivial dependency on Shogun machine learning toolbox 
(minimal version is 2.1.0).

API
---

We provide an interface based on the method chaining technique. The chain starts from the call 
of the `initialize()` method and followed with the `withParameters(const ParametersSet&)` call 
which is used to provide parameters like the method to use and its settings. The provided 
argument is formed with the following syntax:

	(keyword1=value1, keyword2=value2)

Keywords are defined in the `tapkee::keywords` namespace. Currently, the following keywords
are defined: `method`, `eigen_method`, `neighbors_method`, `num_neighbors`, `target_dimension`,
`diffusion_map_timesteps`, `gaussian_kernel_width`, `max_iteration`, `spe_global_strategy`, 
`spe_num_updates`, `spe_tolerance`, `landmark_ratio`, `nullspace_shift`, `klle_shift`, 
`check_connectivity`, `fa_epsilon`, `progress_function`, `cancel_function`, `sne_perplexity`,
`sne_theta`. See the documentation for their detailed meaning.

As an example, if you want to use the Isomap algorithm with the 
number of neighbors set to 15:

	initialize().withParameters((method=Isomap,num_neighbors=15))

Please note that the inner parentheses are necessary as it uses the 
overloaded comma operator which appears to be ambiguous in this case.

Next, with initialized parameters you may either embed the provided matrix with:

	initialize().withParameters((method=Isomap,num_neighbors=15)).
	            .embedUsing(matrix);

Or provide callbacks (kernel, distance and features) using any combination 
of the `withKernel(KernelCallback)`, `withDistance(DistanceCallback)` and 
`withFeatures(FeaturesCallback)` member functions:

	initialize().withParameters((method=Isomap,num_neighbors=15))
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

	TapkeeOutput output = initialize()
	    .withParameters((method=Isomap,num_neighbors=15))
	    .embedUsing(matrix);

	TapkeeOutput output = initialize()
	    .withParameters((method=Isomap,num_neighbors=15))
	    .withDistance(distance_callback)
	    .embedUsing(indices);

	TapkeeOutput output = initialize()
	    .withParameters((method=Isomap,num_neighbors=15))
	    .withDistance(distance_callback)
	    .embedRange(indices.begin(),indices.end());

Minimal example
---------------

A minimal working example of a program that uses the library is:

	#include <tapkee.hpp>
	#include <callback/dummy_callbacks.hpp>

	using namespace std;
	using namespace tapkee;
	using namespace tapkee::keywords;

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

		TapkeeOutput output = initialize() 
		   .withParameters((method=MultidimensionalScaling,target_dimension=1))
		   .withDistance(d)
		   .embedUsing(indices);

		cout << output.embedding.transpose() << endl;
		return 0;
	}

Integration
-----------

There are a few issues related to including the Tapkee library to your code. First, if your library 
already includes Eigen3 (and only if) - you might need to let Tapkee 
know about that with the following define:

`#define TAPKEE_EIGEN_INCLUDE_FILE <path/to/your/eigen/include/file.h>`

Please note that if you don't include Eigen3 in your project there is no need to define that variable -
in this case Eigen3 will be included by Tapkee. This issue comes from the need of including the Eigen3 library
only once when using some specific parameters (like debug and extensions).

If you are able to use less restrictive licenses (such as LGPLv3) you may define 
the following variable:

- `TAPKEE_USE_LGPL_COVERTREE` to use Covertree code by John Langford.

When compiling your software that includes Tapkee be sure Eigen3 headers are in include path and your code
is linked against ARPACK library (-larpack key for g++ and clang++).

For an example of integration you may check 
[Tapkee adapter in Shogun](https://github.com/shogun-toolbox/shogun/blob/master/src/shogun/lib/tapkee/tapkee_shogun.cpp). 

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
before including [defines header](https://github.com/lisitsyn/tapkee/tree/master/include/tapkee_defines.hpp).

If you use some non-standard STL-compatible realization of vector, map and pair you may redefine them
with `TAPKEE_INTERNAL_VECTOR`, `TAPKEE_INTERNAL_PAIR`, `TAPKEE_INTERNAL_MAP` 
(they are set to std::vector, std::pair and std::map by default otherwise).

You may define `TAPKEE_USE_FIBONACCI_HEAP` or `TAPKEE_USE_PRIORITY_QUEUE` to select which
data structure should be used in the shortest paths computing algorithm. By default 
a priority queue is used.

Other properties can be loaded from some provided header file using `#define TAPKEE_CUSTOM_PROPERTIES`. Currently
such file should define only one variable - `COVERTREE_BASE` which defines the base of the CoverTree (default is 1.3).

Command line application
-----------

Tapkee comes with a sample application which can be used to construct
low-dimensional representations of dense feature matrices. For more information on 
its usage please run:

`./bin/tapkee_cli -h`

The application takes plain ASCII file containing dense matrix (each vector is a column and each
line contains values of some feature). The output of the application is stored into the provided
file in the same format (each line is feature).

To compile the application please use [CMake](http://cmake.org/). The workflow of compilation 
Tapkee with CMake is usual. When using Unix-based
systems you may use the following command to compile the Tapkee application:

`mkdir build && cd build && cmake [definitions] .. && make`

There are a few cases when you'd want to put some definitions:

- To enable unit-tests compilation add to `-DBUILD_TESTS=1` to `[definitions]` when building. Please note that 
  building unit-tests require googletest. If you are running Ubuntu you may install `libgtest-dev` package for that. 
  Otherwise, if you have gtest sources around you may provide them as `-DGTEST_SOURCE_DIR` and `-DGTEST_INCLUDES_DIR`.
  If may also download gtest with the following command: 

	wget http://googletest.googlecode.com/files/gtest-1.6.0.zip && 
	unzip -q gtest-1.6.0.zip && cd gtest-1.6.0 && cmake . && 
	make && cd .. && rm gtest-1.6.0.zip
  
  Downloaded sources will be used by Tapkee on build.
  To run tests use `make test` command (or better 'ctest -VV').

- To let make script store test coverage information using GCOV and 
  add a target for output test coverage in HTML with LCOV add the `-DUSE_GCOV=1` flag to `[definitions]`.

- To enable precomputation of kernel/distance matrices which can speed-up algorithms (but requires much more memory) add
  `-DPRECOMPUTED=1` to `[definitions]` when building.

- To build application without parts licensed by LGPLv3 use `-DGPL_FREE=1` definition.

The library requires Eigen3 to be available in your path. The ARPACK library is also highly 
recommended to achieve best performance. On Debian/Ubuntu these packages can be installed with 

	sudo apt-get install libeigen3-dev libarpack2-dev

If you are using Mac OS X and Macports you can install these packages with 

	sudo port install eigen3 && sudo port install arpack`

In case you want to use some non-default 
compiler use `CC=your-C-compiler CXX=your-C++-compiler cmake [definitions] ..` when running cmake.

Need help?
----------

If you need any help or advice don't hesitate to send [an email](mailto://lisitsyn.s.o@gmail.com) or 
fire [an issue at github](https://github.com/lisitsyn/tapkee/issues/new).

Supported platforms
-------------------

Tapkee is tested to be fully functional on Linux (ICC, GCC, Clang compilers) 
and Mac OS X (GCC and Clang compilers). It also compiles under Windows natively
(MSVS 2012 compiler) with a few known issues. In general, Tapkee uses no platform 
specific code and should work on other systems as well. Please 
[let us know](mailto://lisitsyn.s.o@gmail.com) if you have successfully compiled 
or have got any issues on any other system not listed above.

Supported dimension reduction methods
-------------------------------------

Tapkee provides implementations of the following dimension reduction methods (urls to descriptions provided):

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
* PCA and randomized PCA
* Kernel PCA (kPCA)
* Random projection
* Factor analysis
* t-SNE
* Barnes-Hut-SNE

Licensing
---------

The library is distributed under the [BSD 3-clause](https://github.com/lisitsyn/tapkee/tree/master/LICENSE) license.

Exceptions are:

- [Barnes-Hut-SNE code](https://github.com/lisitsyn/tapkee/tree/master/include/external/barnes_hut_sne/) by Laurens van der Maaten which
  is distributed under the BSD 4-clause license.

- [Covertree code](https://github.com/lisitsyn/tapkee/tree/master/include/neighbors/covertree.hpp) by John Langford and Dinoj Surendran 
  which is distributed under the [LGPLv3 license](https://github.com/lisitsyn/tapkee/tree/master/LGPL-LICENSE).

- [EZOptionsParser](https://github.com/lisitsyn/tapkee/tree/master/src/ezoptionparser.hpp) by Remik Ziemlinski which is distributed 
  under the [MIT license](https://github.com/lisitsyn/tapkee/tree/master/MIT-LICENSE).
