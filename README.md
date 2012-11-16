Tapkee: an efficient dimension reduction toolbox
================================================

Tapkee is a C++ template library for dimensionality reduction with some bias on 
spectral methods. The Tapkee origins from the code developed during GSoC 2011 as 
the part of the Shogun machine learning toolbox. The project aim is to provide 
standalone efficient and flexible library for dimensionality reduction.

Tapkee uses highly effective Eigen3 template linear algebra library and 
optionally makes use of the ARPACK and the Feast eigensolvers.

Interface
---------

This toolbox considers dimensionality reduction algorithms separate from data representation.
To let user choose how to handle his data we provide callback interface essentially based
on three functions: kernel function (similarity), distance function (dissimilarity) and 
feature vector access function. It is worth to notice that most of methods use either
kernel or distance when all linear (projective) methods require access to feature vector.

Callback interface enables user to reach great flexibility: ability to set up some caching strategy,
lazy initialization of resources and various more. As an example we provide simple dense feature 
matrices callback set.

Supported platforms
-------------------

Tapkee is tested to be functional on Linux and Mac OS X platforms using GCC and Clang compilers.

Methods
-------

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
* Neighborhood Component Analysis (NCA)

The following methods are under development:

* Coordinated Factor Analysis (CFA)
* Maximum Variance Unfolding and landmark Maximum Variance Unfolding (MVU)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
