Tapkee is a C++ template library for dimensionality reduction with some bias on 
spectral methods. The Tapkee origins from the code developed during 
[GSoC 2011](http://www.google-melange.com/gsoc/homepage/google/gsoc2011) as 
the part of the [Shogun machine learning toolbox](https://github.com/shogun-toolbox/shogun). 
The project aim is to provide efficient and flexible standalone library for 
dimensionality reduction which can be easily integrated to existing codebases.
Tapkee leverages capabilities of effective [Eigen3 linear algebra library](http://eigen.tuxfamily.org) and 
optionally makes use of the [ARPACK eigensolver](http://www.caam.rice.edu/software/ARPACK/). To achieve 
greater flexibility we provide a callback interface which decouples dimension reduction algorithms from
the data representation and storage schemes (see Callback interface section).

The library is distributed under permissive [BSD 3-clause license](LICENSE) 
(except some parts that are distributed under other 
open sources licenses, see Licensing section of this document).

API
---

The main entry point of Tapkee is the [embed](https://github.com/lisitsyn/tapkee/tree/master/include/tapkee.hpp) 
function (see [the documentation](http://www.tapkee-library.info/doxygen/html/index.html) for more details) that
returns embedding for provided data and a function that can be used to project data out of the sample (if such
function is implemented). This function takes two random access iterators (that denote begin and end of the data),
three callbacks (kernel callback, distance callback and feature vector access callback) and parameters map.

In the simplest case `begin` and `end` are set to the corresponding iterators of some `vector<tapkee::IndexType>` 
filled with a range of values from 0 to N-1 where N is the number of vectors to embed. 

Callbacks, such as kernel functor (that computes similarity), distance functor (that computes dissimilarity) and 
dense feature vector access functor (that computes required feature vector), are used by the library to access the 
data. Such interface is motivated by great flexibility it provides (custom caching, precomputing, maybe even network access). 
As an example we provide [a simple callback set](https://github.com/lisitsyn/tapkee/tree/master/include/callback/eigen_callbacks.hpp)
for dense feature matrices out-of-the-box. If you are working with precomputed kernel and distance matrices you may find
[precomputed callbacks](https://github.com/lisitsyn/tapkee/tree/master/include/callback/precomputed_callbacks.hpp) 
useful. It is worth to note that most of methods use either kernel or distance while all linear (projective) methods require 
access to feature vector. For example, to use the Locally Linear Embedding algorithm it is enough to provide a kernel
callback; the Multidimensional Scaling algorithm requires only a distance callback and PCA requires only a feature
vector access callback. Full set of callbacks (all three callbacks) makes possible to use all the implemented methods.

Parameters map should contain all the required parameters as values with keys as 
[TAPKEE\_PARAMETERS](https://github.com/lisitsyn/tapkee/blob/master/include/tapkee_defines.hpp#L61). If 
some parameter is missed in the map the library throws an exception with information about missed parameter.
You may check which parameters do you have to set in the documentation of parameters and methods or in 
[implementations](https://github.com/lisitsyn/tapkee/blob/master/include/tapkee_methods.hpp)
where parameters are obtained using the `PARAMETER` macro. 

For example to run the Locally Linear Embedding algorithm you might need to populate the parameters map with the following code:
`
tapkee::ParametersMap parameters;
parameters[tapkee::REDUCTION_METHOD] = LOCALLY_LINEAR_EMBEDDING;
parameters[tapkee::NEIGHBORS_METHOD] = COVER_TREE;
parameters[tapkee::EIGEN_EMBEDDING_METHOD] = ARPACK;
parameters[tapkee::TARGET_DIMENSION] = static_cast<tapkee::IndexType>(2);
parameters[tapkee::NUMBER_OF_NEIGHBORS] = static_cast<tapkee::IndexType>(20);
`

Integration issues
------------------

There are a few issues related to including the Tapkee library. First, if your library already includes 
Eigen3 (and only if) - you might need to let Tapkee know about that with the following define:

`#define TAPKEE_EIGEN_INCLUDE_FILE <path/to/your/eigen/include/file.h>`

Please note that if you don't include Eigen3 in your project there is no need to define that variable -
Eigen3 will be included by Tapkee in this case. This issue comes from the need of including the Eigen3 library
only once when using some specific parameters (like debug and extensions).

It is also required to identify your custom callback functors using the following macroses:

`TAPKEE_CALLBACK_IS_KERNEL(your_kernel_callback)`

`TAPKEE_CALLBACK_IS_DISTANCE(your_distance_callback)`

If you are able to use less restrictive licenses (such as LGPLv3) you could define 
the following variable:

- `TAPKEE_USE_LGPL_COVERTREE` to use Covertree code by John Langford.

When compiling your software that includes Tapkee be sure Eigen3 headers are in include path and your code
is linked against ARPACK library (-larpack key for g++ and clang++).

For an example of integration you may check 
[Tapkee adapter in Shogun](https://github.com/shogun-toolbox/shogun/blob/master/src/shogun/lib/tapkee/tapkee_shogun.cpp). 

When working with installed headers you may check which version of the library 
do you have with checking the values of `TAPKEE_MAJOR_VERSION` and `TAPKEE_MINOR_VERSION` defines.

We welcome any integration so please contact authors if you have got any questions. If you have 
successfully used the library please also let authors know about that - mentions of any
applications are very appreciated.

Customization
-------------

Tapkee is being developed to be highly customizable with preprocessor definitions.

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

Application
-----------

Tapkee comes with a sample application which can be used to construct
low-dimensional representations of dense feature matrices. For more information on 
its usage please run:

`./tapkee -h`

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
  `wget http://googletest.googlecode.com/files/gtest-1.6.0.zip && 
   unzip -q gtest-1.6.0.zip && cd gtest-1.6.0 && cmake . && 
   make && cd .. && rm gtest-1.6.0.zip`
  Downloaded sources will be used by Tapkee on build.
  To run tests use `make test` command (or better 'ctest -VV').

- To let make script store test coverage information using GCOV and 
  add a target for output test coverage in HTML with LCOV add the `-DUSE_GCOV=1` flag to `[definitions]`.

- To enable precomputation of kernel/distance matrices which can speed-up algorithms (but requires much more memory) add
  `-DPRECOMPUTED=1` to `[definitions]` when building.

- To build application without parts licensed by LGPLv3 use `-DGPL_FREE=1` definition.

The library requires Eigen3 to be available in your path. The ARPACK library is also highly 
recommended to achieve best performance. On Debian/Ubuntu these packages can be installed with 

`sudo apt-get install libeigen3-dev libarpack2-dev`

If you are using Mac OS X and Macports you can install these packages with 

`sudo port install eigen3 && sudo port install arpack`

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

* [Locally Linear Embedding and Kernel Locally Linear Embedding (LLE/KLLE)](http://www.tapkee-library.info/methods/lle.html)
* [Neighborhood Preserving Embedding (NPE)](http://www.tapkee-library.info/methods/npe.html)
* [Local Tangent Space Alignment (LTSA)](http://www.tapkee-library.info/methods/ltsa.html)
* [Linear Local Tangent Space Alignment (LLTSA)](http://www.tapkee-library.info/methods/lltsa.html)
* [Hessian Locally Linear Embedding (HLLE)](http://www.tapkee-library.info/methods/hlle.html)
* [Laplacian eigenmaps](http://www.tapkee-library.info/methods/le.html)
* [Locality Preserving Projections](http://www.tapkee-library.info/methods/lpp.html)
* [Diffusion map](http://www.tapkee-library.info/methods/dm.html)
* [Isomap and landmark Isomap](http://lisitsyn.github.com/tapkee/methods/isomap.html)
* [Multidimensional scaling and landmark Multidimensional scaling (MDS/lMDS)](http://www.tapkee-library.info/methods/mds.html)
* [Stochastic Proximity Embedding (SPE)](http://www.tapkee-library.info/methods/spe.html)
* [PCA and randomized PCA](http://www.tapkee-library.info/methods/pca.html)
* [Kernel PCA (kPCA)](http://www.tapkee-library.info/methods/kpca.html)
* [Random projection](http://www.tapkee-library.info/methods/rp.html)
* [Factor analysis](http://www.tapkee-library.info/methods/fa.html)
* [t-SNE](http://www.tapkee-library.info/methods/tsne.html)
* [Barnes-Hut-SNE](http://www.tapkee-library.info/methods/tsne.html)

Licensing
---------

The library is distributed under the [BSD 3-clause](https://github.com/lisitsyn/tapkee/tree/master/LICENSE) license.

Exceptions are:

- [Barnes-Hut-SNE code](https://github.com/lisitsyn/tapkee/tree/master/include/external/barnes_hut_sne/) by Laurens van der Maaten which
  is distributed under the BSD 4-clause license.

- [Covertree code](https://github.com/lisitsyn/tapkee/tree/master/include/neighbors/covertree.hpp) by John Langford and Dinoj Surendran 
  which is distributed under the [LGPLv3 license](LGPL-LICENSE).

- [Any type](https://github.com/lisitsyn/tapkee/tree/master/include/utils/any.hpp) by Christopher Diggins which is distributed under 
  the [Boost v.1.0 license](http://www.boost.org/LICENSE_1_0.txt).

- [EZOptionsParser](https://github.com/lisitsyn/tapkee/tree/master/src/ezoptionparser.hpp) by Remik Ziemlinski which is distributed 
  under the [MIT license](MIT-LICENSE).
