Isomap
------

The Isomap algorithm can be considered as an modification of the classic Multidimensional
Scaling algorithm. The algorithm itself is consisted of the following steps:

* For each feature vector $ x \in X $ find $ k $ its nearest neighbors and
  construct the sparse neighborhood graph.

* Compute squared distances matrix $ D $ such as $ D\_{i,j} = d^2(x\_i,x\_j) $.
 
* Relax distances with shortest (so-called geodesic) distances on the sparse 
  neighborhood graph (e.g. with sparse Dijkstra algorithm).

* Center the matrix $ D $ with subtracting row mean, column mean and adding the 
  grand mean. Multiply $ D $ element-wise with $ -0.5 $.

* Compute embedding with the $ t$ eigenvectors that correspond to the 
  largest eigenvalues of the matrix $ D $; normalize these vectors
  with dividing each eigenvectors with square root of its corresponding
  eigenvalue. Form the final embedding with eigenvectors as rows and projected
  feature vectors as columns.

References
----------

* Tenenbaum, J. B., De Silva, V., & Langford, J. C. (2000). 
  [A global geometric framework for nonlinear dimensionality reduction](http://www.robots.ox.ac.uk/~az/lectures/ml/tenenbaum-isomap-Science2000.pdf)
