Classic multidimensional scaling 
--------------------------------

The classic multidimensional scaling algorithm is probably the simplest dimensionality
reduction algorithm which reduced data in an attempt to keep pairwise distances the same.
The algorithm itself is:

* For a given set of vectors $ X = x\_1, x\_2, \dots, x\_N $ compute 
  the pairwise distances matrix $ D $ such that $ D\_{i,j} = d(x\_i,x\_j) $,
 
* Square each element of the distances matrix $ D $ and center the matrix
  with subtracting row mean, column mean and adding the grand mean.

* Compute embedding with the $ t$ eigenvectors that correspond to the 
  largest eigenvalues of the matrix $ D $; normalize these vectors
  with dividing each eigenvectors with square root of its corresponding
  eigenvalue. Form the final embedding with eigenvectors as rows and projected
  feature vectors as columns.
