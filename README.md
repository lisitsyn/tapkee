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

The Tapkee library is designed to be easily integrated to other codebases. 
For examples of such integration see [Shogun machine learning toolbox](https://github.com/shogun-toolbox/shogun).

Application
-----------

Tapkee comes with a sample application used to embed dense representations, for more information run

`./tapkee_app -h`


Need help?
----------

If you need any help or advice don't hesitate to send [an email](mailto://lisitsyn.s.o@gmail.com "Send mail
to Sergey Lisitsyn") or fire [an issue at github](https://github.com/lisitsyn/tapkee/issues/new "New Tapkee Issue").

Supported platforms
-------------------

Tapkee is tested to be fully functional on Linux (ICC, GCC, Clang compilers) 
and Mac OS X (GCC and Clang compilers).

Supported dimension reduction methods
-------------------------------------

Tapkee provides implementations of the following dimension reduction methods:

* Locally Linear Embedding / Kernel Locally Linear Embedding (LLE/KLLE)
* Neighborhood Preserving Embedding (NPE)
* Local Tangent Space Alignment / Kernel Local Tangent Space Alignment (LTSA/KLTSA)
* Linear Local Tangent Space Alignment (LLTSA)
* Hessian Locally Linear Embedding (HLLE)
* Diffusion map
* Laplacian eigenmaps
* Locality Preserving Projections (LPP)
* Multidimensional scaling and landmark Multidimensional scaling (MDS/lMDS)
* Isomap and landmark Isomap 
* Stochastic Proximity Embedding (SPE)
* Kernel PCA (kPCA)

The following methods are under development:

* Coordinated Factor Analysis (CFA)
* Neighborhood Component Analysis (NCA)
* Maximum Variance Unfolding and landmark Maximum Variance Unfolding (MVU)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
