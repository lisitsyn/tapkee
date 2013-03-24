Hessian Locally Linear Embedding
--------------------------------

Just like the Local Tangent Space Alignment, the Hessian Locally Linear Embedding algorithm is very 
similar to the Locally Linear Embedding algorithm.

Given a set of feature vectors \(X = \{ x_1, x_2, \dots x_N \} \) the HLLE algorithm proposes to perform the following steps:

* Identify nearest neighbors. For each \( x \in X \) identify its \( k \) nearest 
  neighbors, i.e. a set \( \mathcal{N}_x \) of \( k \) feature vectors such that
  \begin{equation} \arg\min_{\mathcal{N}_x} \sum_{x_n \in \mathcal{N}_x}\| x - x_n \|_2^2 \label{nearest_neighbors}\end{equation}
  
* Analyze hessian of each local patch. For each \( x \in X \) compute the 
  Gram matrix \( G \) of its neighbors such that \( G_{i,j} = (\mathcal{N}_x^{i} , \mathcal{N}_x^{j}) \) 
  and center  it. Compute its \( t \) (the number of required features) eigenvectors \( v_1, \dots v_t \).
  Construct hessian approximating matrix 
  \[ Y = \begin{bmatrix} 1_k & v_1 & \dots & v_t & v_1\cdot v_1 & \dots & v_1\cdot v_t & \dots \end{bmatrix}, \]
  where \( \cdot : X \times X \to X \) denotes coefficient-wise product. Normalize columns of the matrix \( Y \) and then compute matrix 
  \[ Q = Y Y^{T}\]
  and put it to the sparse alignment matrix \( L \) (initially set by zeroes) using the following procedure:
  \[ L \leftarrow L + Q. \]
  </li>

* Embedding through eigendecomposition. To obtain \( t \) features (coordinates) of embedded vectors 
  solve the partial eigenproblem 
  \[ L f = \lambda f, \]
  for smallest eigenvalues \( \lambda_1, \dots, \lambda_t, \lambda_{t+1} \) and its corresponding 
  eigenvectors \( f_1, \dots, f_t, f_{t+1} \). Drop the smallest eigenvalue \( \lambda_1 \sim 0 \) 
  (with its corresponding eigenvector) and form embedding matrix such that 
  \(i\)-th coordinate (\( i=1,\dots,N \)) of \(j\)-th eigenvector 
  (\( j=1,\dots,t\) ) corresponds to \(j\)-th coordinate of projected \(i\)-th vector.

References
----------

* Donoho, D., & Grimes, C. (2003). 
[Hessian eigenmaps: new locally linear embedding techniques for high-dimensional data](http://www-stat.stanford.edu/~donoho/Reports/2003/HessianEigenmaps.pdf")
