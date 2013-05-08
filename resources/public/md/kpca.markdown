Kernel Principal Component Analysis
-----------------------------------

The Kernel Principal Component Analysis algorithm is a generalization of the PCA algorithm.
The algorithm performs the following steps

* Compute the kernel matrix $K$ such that $K\_{i,j} = k(x\_i,x\_j)$ where $k : X \times X \to \mathbb{R}$ is 
  a Mercer kernel function and $X$ is a set of feature vectors $\\{ x\_1, x\_2, \dots, x\_N \\}$

* Center the matrix $K$ with subtracting row mean, column mean and adding the grand mean.

* Compute embedding with the $ t$ eigenvectors that correspond to the 
  largest eigenvalues of the matrix $ D $; normalize these vectors
  with dividing each eigenvectors with square root of its corresponding
  eigenvalue. Form the final embedding with eigenvectors as rows and projected
  feature vectors as columns.

References
----------

* Schölkopf, B., Smola, A., & Müller, K. R. (1997). 
  Kernel principal component analysis
