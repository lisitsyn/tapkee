Tapkee: an efficient dimension reduction library
================================================

Tapkee is a C++ template library for dimensionality reduction with some bias on 
spectral methods. The Tapkee origins from the code developed during 
[GSoC 2011](http://www.google-melange.com/gsoc/homepage/google/gsoc2011) as 
the part of the [Shogun machine learning toolbox](https://github.com/shogun-toolbox/shogun). 
The project aim is to provide efficient and flexible standalone library for 
dimensionality reduction which can be easily integrated to existing codebases.
Tapkee leverages capabilities of effective [Eigen3 linear algebra library](http://eigen.tuxfamily.org) and 
optionally makes use of the [ARPACK eigensolver](http://www.caam.rice.edu/software/ARPACK/). To achieve 
great flexibility we provide a callback interface which decouples dimension reduction algorithms from
the data representation and storage schemes (see Callback interface section).

Contributions are very encouraged as we distribute our software under permissive 
[BSD 3-clause license](LICENSE) (except some parts that are distributed under other 
open sources licenses, see Licensing section of this document).

To achieve code quality we employ [googletest](https://code.google.com/p/googletest/) as a testing
framework (tests can be found [here](test/gtest)), [valgrind](http://valgrind.org/) for dynamic 
analysis and [clang static analyzer](http://clang-analyzer.llvm.org/) as a tool for static code 
analysis. We are happy to use [Travis](https://travis-ci.org) as a continuous integration 
platform. The build status is:

[![Build Status](https://travis-ci.org/lisitsyn/tapkee.png)](https://travis-ci.org/lisitsyn/tapkee)

Callback interface
------------------

To achieve greater flexibility, the library decouples algorithms from data representation.
To let user choose how to handle his data we provide callback interface essentially based
on three functions: kernel function (similarity), distance function (dissimilarity) and 
dense feature vector access function. It is worth to notice that most of methods use either
kernel or distance while all linear (projective) methods require access to feature vector. 
Full set of callbacks (all three callbacks) makes possible to use all implemented methods.

Callback interface enables user to reach great flexibility: ability to set up some caching strategy,
lazy initialization of resources and various more. As an example we provide 
[simple callback set](https://github.com/lisitsyn/tapkee/tree/master/tapkee/callback/eigen_callbacks.hpp)
for dense feature matrices out-of-the-box. If you are able to precompute kernel and distance matrices you may find
[precomputed callbacks](https://github.com/lisitsyn/tapkee/tree/master/tapkee/callback/precomputed_callbacks.hpp) 
useful.

It is required to identify your callback functors using the following macroses:

`TAPKEE_CALLBACK_IS_KERNEL(your_kernel_callback)`

`TAPKEE_CALLBACK_IS_DISTANCE(your_distance_callback)`

Out-of-the-box callbacks are already 'identified' - no need to use any macroses for them.

Integration with other libraries
--------------------------------

The main entry point of Tapkee is [embed](https://github.com/lisitsyn/tapkee/tree/master/tapkee/tapkee.hpp) 
method (see the documentation for more details).

If your library includes Eigen3 (and only if) at some point - 
let the Tapkee know about that with the following define:

`#define TAPKEE_EIGEN_INCLUDE_FILE <path/to/your/eigen/include/file.h>`

Please note that if you don't include Eigen3 in your project there is no need to define that variable -
Eigen3 will be included by Tapkee in this case.

If you are able to use less restrictive licenses (such as LGPLv3) you could define 
the following variable:

- `TAPKEE_USE_LGPL_COVERTREE` to use Covertree code by John Langford.

When compiling your software that includes Tapkee be sure Eigen3 headers are in include path and your code
is linked against ARPACK library (-larpack key for g++ and clang++).

For an example of integration you may check 
[Tapkee adapter in Shogun](https://github.com/shogun-toolbox/shogun/blob/master/src/shogun/lib/tapkee/tapkee_shogun.cpp). 

To control the flow you may also provide callbacks that track progress
and indicate if computations were cancelled (`tapkee::PROGRESS_FUNCTION` and `tapkee::CANCEL_FUNCTION` keys).

We welcome any integration so please contact authors if you have got any questions. If you have 
successfully used the library please also let authors know about that - mentions of any
applications are very appreciated.

Customization
-------------

Tapkee is supposed to be highly customizable with preprocessor definitions.

If you want to use float as numeric type (default is double) you may do 
that with definition of `#define TAPKEE_CUSTOM_NUMTYPE float` 
before including [defines header](https://github.com/lisitsyn/tapkee/tree/master/tapkee/tapkee_defines.hpp).

If you use some non-standard STL-compatible realization of vector, map and pair you may redefine them
with `TAPKEE_INTERNAL_VECTOR`, `TAPKEE_INTERNAL_PAIR`, `TAPKEE_INTERNAL_MAP` 
(they are set to std::vector, std::pair and std::map by default).

You may define `TAPKEE_USE_FIBONACCI_HEAP` or `TAPKEE_USE_PRIORITY_QUEUE` to select which
data structure should be used for shortest paths computing. By default a priority queue is used.

Other properties can be loaded from some provided header file using `#define TAPKEE_CUSTOM_PROPERTIES`. Currently
such file should define the variable `COVERTREE_BASE` which is base of the CoverTree to be used (default is 1.3).

Application
-----------

Tapkee comes with a sample application which can be used to construct
low-dimensional representations of feature matrices. For more information on its usage please run:

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
  `wget http://googletest.googlecode.com/files/gtest-1.6.0.zip && unzip -q gtest-1.6.0.zip && cd gtest-1.6.0 && cmake . && make && cd .. && rm gtest-1.6.0.zip`.
  Downloaded sources will be used by Tapkee on build.
  To run tests use `make test` command (or better 'ctest -VV').

- To let make script store test coverage information using GCOV and 
  add a target for output test coverage in HTML with LCOV add the `-DUSE_GCOV=1` flag to `[definitions]`.

- To enable precomputation of kernel/distance matrices which can speed-up algorithms (but requires much more memory) add
  `-DPRECOMPUTED=1` to `[definitions]` when building.

- To build application without parts licensed by GPLv3 and LGPLv3 use `-DGPL_FREE=1` definition.

The compilation requires Eigen3 to be available in your path. The ARPACK library is also highly recommended. 
On Ubuntu Linux these packages can be installed with 

`sudo apt-get install libeigen3-dev libarpack2-dev`

If you are using Mac OS X and Macports you can install these packages with 

`sudo port install eigen3 && sudo port install arpack`

In case you want to use some non-default 
compiler use `CC=your-C-compiler CXX=your-C++-compiler cmake` when running cmake.

Need help?
----------

If you need any help or advice don't hesitate to send [an email](mailto://lisitsyn.s.o@gmail.com "Send mail
to Sergey Lisitsyn") or fire [an issue at github](https://github.com/lisitsyn/tapkee/issues/new "New Tapkee Issue").

Supported platforms
-------------------

Tapkee is tested to be fully functional on Linux (ICC, GCC, Clang compilers) 
and Mac OS X (GCC and Clang compilers). It also compiles under Windows (MSVS 2012 compiler)
but wasn't properly tested yet. In general, Tapkee uses no platform specific code 
and should work on other systems as well. Please [let us know](mailto://lisitsyn.s.o@gmail.com) 
if you have successfully compiled or have got issues on any other system not listed above.

Supported dimension reduction methods
-------------------------------------

Tapkee provides implementations of the following dimension reduction methods (urls to descriptions provided):

* [Locally Linear Embedding and Kernel Locally Linear Embedding (LLE/KLLE)](http://lisitsyn.github.com/tapkee/methods/lle.html)
* [Neighborhood Preserving Embedding (NPE)](http://lisitsyn.github.com/tapkee/methods/npe.html)
* [Local Tangent Space Alignment (LTSA)](http://lisitsyn.github.com/tapkee/methods/ltsa.html)
* [Linear Local Tangent Space Alignment (LLTSA)](http://lisitsyn.github.com/tapkee/methods/lltsa.html)
* [Hessian Locally Linear Embedding (HLLE)](http://lisitsyn.github.com/tapkee/methods/hlle.html)
* [Laplacian eigenmaps](http://lisitsyn.github.com/tapkee/methods/laplacian_eigenmaps.html)
* [Locality Preserving Projections](http://lisitsyn.github.com/tapkee/methods/lpp.html)
* [Diffusion map](http://lisitsyn.github.com/tapkee/methods/diffusion_map.html)
* [Isomap and landmark Isomap](http://lisitsyn.github.com/tapkee/methods/isomap.html)
* [Multidimensional scaling and landmark Multidimensional scaling (MDS/lMDS)](http://lisitsyn.github.com/tapkee/methods/mds.html)
* [Stochastic Proximity Embedding (SPE)](http://lisitsyn.github.com/tapkee/methods/spe.html)
* [PCA and randomized PCA](http://lisitsyn.github.com/tapkee/methods/pca.html)
* [Kernel PCA (kPCA)](http://lisitsyn.github.com/tapkee/methods/kpca.html)
* [Random projection](http://lisitsyn.github.com/tapkee/methods/ra.html)
* [Factor analysis](http://lisitsyn.github.com/tapkee/methods/fa.html)
* [t-SNE](http://lisitsyn.github.com/tapkee/method/tsne.html)
* [Barnes-Hut-SNE](htpp://lisitsyn.github.com/tapkee/method/barnes_hut_sne.html)

Licensing
---------

The library is distributed under the [BSD 3-clause](LICENSE) license.

Exceptions are:

- [Barnes-Hut-SNE code](tapkee/external/barnes_hut_sne/) by Laurens van der Maaten which
  is distributed under the BSD 4-clause license.

- [Covertree code](tapkee/neighbors/covertree.hpp) by John Langford and Dinoj Surendran 
  which is distributed under the [LGPLv3 license](LGPL-LICENSE).

- [Any type](tapkee/utils/any.hpp) by Christopher Diggins which is distributed under 
  the [Boost v.1.0 license](http://www.boost.org/LICENSE_1_0.txt).

- [EZOptionsParser](src/ezoptionparser.hpp) by Remik Ziemlinski which is distributed 
  under the [MIT license](MIT-LICENSE).
