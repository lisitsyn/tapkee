Hessian Locally Linear Embedding
--------------------------------

Just like the Local Tangent Space Alignment, the Hessian Locally Linear Embedding algorithm is very 
similar to the Locally Linear Embedding algorithm.

Given a set of feature vectors $X = \{ x\_1, x\_2, \dots x\_N \} $ the HLLE algorithm proposes to perform the following steps:

* Identify nearest neighbors. For each $ x \in X $ identify its $ k $ nearest 
  neighbors, i.e. a set $ \mathcal{N}\_x $ of $ k $ feature vectors such that
  $$ \arg\min\_{\mathcal{N}_x} \sum\_{x\_n \in \mathcal{N}\_x}\| x - x\_n \|\_2^2 $$
  
* Analyze hessian of each local patch. For each $ x \in X $ compute the 
  Gram matrix $ G $ of its neighbors such that $ G\_{i,j} = (\mathcal{N}\_x^{i} , \mathcal{N}\_x^{j}) $ 
  and center it. Compute its $ t $ (the number of required features) eigenvectors $ v\_1, \dots v\_t $.
  Construct hessian approximating matrix 
  $$ Y = \begin{bmatrix} 1\_k & v\_1 & \dots & v\_t & v\_1\cdot v\_1 & \dots & v\_1\cdot v\_t & \dots \end{bmatrix}, $$
  where $ \cdot : X \times X \to X $ denotes coefficient-wise product. Normalize columns of the matrix $ Y $ and then compute matrix 
  $$ Q = Y Y^{T}$$
  and put it to the sparse alignment matrix $ L $ (initially set by zeroes) using the following procedure:
  $$ L \leftarrow L + Q. $$
  </li>

* Embedding through eigendecomposition. To obtain $ t $ features (coordinates) of embedded vectors 
  solve the partial eigenproblem 
  $$ L f = \lambda f, $$
  for smallest eigenvalues $ \lambda\_1, \dots, \lambda\_t, \lambda\_{t+1} $ and its corresponding 
  eigenvectors $ f\_1, \dots, f\_t, f\_{t+1} $. Drop the smallest eigenvalue $ \lambda\_1 \sim 0 $ 
  (with its corresponding eigenvector) and form embedding matrix such that 
  $i$-th coordinate ($ i=1,\dots,N $) of $j$-th eigenvector 
  ($ j=1,\dots,t$ ) corresponds to $j$-th coordinate of projected $i$-th vector.

References
----------

* Donoho, D., & Grimes, C. (2003). 
[Hessian eigenmaps: new locally linear embedding techniques for high-dimensional data](http://www-stat.stanford.edu/~donoho/Reports/2003/HessianEigenmaps.pdf)
