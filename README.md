Tapkee: an efficient dimension reduction toolbox
================================================

Tapkee is a C++ template library for dimensionality reduction with some bias on 
spectral methods. The Tapkee origins from the code developed during GSoC 2011 as 
the part of the [Shogun machine learning toolbox](https://github.com/shogun-toolbox/shogun). 
The project aim is to provide standalone efficient and flexible library for 
dimensionality reduction. We always encourage contributions and distribute our software
under free GPLv3 license.

Tapkee uses highly effective Eigen3 template linear algebra library and 
optionally makes use of the ARPACK eigensolver.

We are happy to use [Travis](https://travis-ci.org) as a continuous integration 
platform [![Build Status](https://travis-ci.org/lisitsyn/tapkee.png)](https://travis-ci.org/lisitsyn/tapkee).

Interface
---------

This toolbox considers dimensionality reduction algorithms separate from data representation.
To let user choose how to handle his data we provide callback interface essentially based
on three functions: kernel function (similarity), distance function (dissimilarity) and 
dense feature vector access function. It is worth to notice that most of methods use either
kernel or distance while all linear (projective) methods require access to feature vector.

Callback interface enables user to reach great flexibility: ability to set up some caching strategy,
lazy initialization of resources and various more. As an example we provide simple callback set
for dense feature matrices.

Integration with other libraries
--------------------------------

The Tapkee library is designed to be easily integrated to other codebases. To use full 
functionality of the Tapkee one should define three functors: kernel, distance and
feature vector access. However it is also possible to construct embeddings with only
kernel or distance callbacks being available - in this case just define non-available
callbacks as some dummies.

It is also required to identify these functors using the following macroses:

`TAPKEE_CALLBACK_IS_KERNEL(your_kernel_callback)`

`TAPKEE_CALLBACK_IS_DISTANCE(your_distance_callback)`

If your library includes Eigen3 at some point - let the Tapkee know about that with the following define:

`#define TAPKEE_EIGEN_INCLUDE_FILE <path/to/your/eigen/include/file.h>`

Please note that if you don't use Eigen3 in your projects there is no need to define that variable, Eigen3 will
be included by Tapkee in this case.

When compiling your software that includes Tapkee be sure Eigen3 headers are in include path and your code
is linked against ARPACK library (-larpack key for g++ and clang++).

For an example of integration see  
[Tapkee adapter in Shogun](https://github.com/shogun-toolbox/shogun/blob/master/src/shogun/lib/tapkee/tapkee_shogun.cpp). 

We welcome any integration so please contact authors if you have got any questions. If you have 
successfully used the library please also let authors know about that - mentions of any 
applications are very appreciated.

Customization
-------------

Tapkee is supposed to be highly customizable with preprocessor definitions.

If you want to use float as numeric type you may do that using definition of `TAPKEE_CUSTOM_NUMTYPE`.

If you use some non-standard STL-compatible realization of vector, map and pair you may redefine them
with `TAPKEE_INTERNAL_VECTOR`, `TAPKEE_INTERNAL_PAIR`, `TAPKEE_INTERNAL_MAP` (std::vector, std::pair, std::map by
default).

Application
-----------

Tapkee comes with a sample application used to embed dense representations, for more information run:

`./tapkee -h`

To compile the application please use CMake. The workflow of compilation with CMake is usual. When using Unix-based
systems you may use the following command to compile the Tapkee application:

`mkdir build && cd build && cmake .. && make`

The compilation requires ARPACK and Eigen3 to be available in your path. On Ubuntu Linux these packages can be installed with 

`sudo apt-get install libeigen3-dev libarpack2-dev`

If you are using Mac OS X and Macports you can install packages with 

`sudo port install eigen3 && sudo port install arpack`

In case you want to use some non-default compiler use `CC=your C compiler CXX=your C++ compiler cmake ..` when using cmake.

Need help?
----------

If you need any help or advice don't hesitate to send [an email](mailto://lisitsyn.s.o@gmail.com "Send mail
to Sergey Lisitsyn") or fire [an issue at github](https://github.com/lisitsyn/tapkee/issues/new "New Tapkee Issue").

Supported platforms
-------------------

Tapkee is tested to be fully functional on Linux (ICC, GCC, Clang compilers) 
and Mac OS X (GCC and Clang compilers). In general, Tapkee uses no platform specific
code and should work on other systems as well. Please [let us know](mailto://lisitsyn.s.o@gmail.com) if you have
successfully compiled or have got issues on any other system not listed above.

Supported dimension reduction methods
-------------------------------------

Tapkee provides implementations of the following dimension reduction methods (urls to descriptions provided):

* [Locally Linear Embedding / Kernel Locally Linear Embedding (LLE/KLLE)](http://lisitsyn.github.com/tapkee/methods/lle.html)
* [Neighborhood Preserving Embedding (NPE)](http://lisitsyn.github.com/tapkee/methods/npe.html)
* [Local Tangent Space Alignment / (LTSA)](http://lisitsyn.github.com/tapkee/methods/ltsa.html)
* [Linear Local Tangent Space Alignment (LLTSA)](http://lisitsyn.github.com/tapkee/methods/lltsa.html)
* [Hessian Locally Linear Embedding (HLLE)](http://lisitsyn.github.com/tapkee/methods/hlle.html)
* [Diffusion map](http://lisitsyn.github.com/tapkee/methods/diffusion_map.html)
* [Laplacian eigenmaps](http://lisitsyn.github.com/tapkee/methods/laplacian_eigenmaps.html)
* [Locality Preserving Projections (LPP)](http://lisitsyn.github.com/tapkee/methods/lpp.html)
* Multidimensional scaling and landmark Multidimensional scaling (MDS/lMDS)
* Isomap and landmark Isomap
* [Stochastic Proximity Embedding (SPE)](http://lisitsyn.github.com/tapkee/methods/spe.html)
* PCA and randomized PCA
* Kernel PCA (kPCA)
* Random projection

The following methods are under development:

* Coordinated Factor Analysis (CFA)
* Neighborhood Component Analysis (NCA)
* Maximum Variance Unfolding and landmark Maximum Variance Unfolding (MVU)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
