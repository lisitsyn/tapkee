Local Tangent Space Alignment
-----------------------------

The Local Tangent Space Alignment algorithm is pretty similar
to the Locally Linear Embedding algorithm.

Given a set of feature vectors \(X = \{ x_1, x_2, \dots x_N \} \) 
the Local Tangent Space Alignment algorithm performs the following steps:

* Identify nearest neighbors. For each \( x \in X \) identify its \( k \) 
  nearest neighbors, i.e. a set \( \mathcal{N}_x \) of \( k \) feature vectors such that
  \begin{equation} \arg\min_{\mathcal{N}_x} \sum_{x_n \in \mathcal{N}_x}\| x - x_n \|_2^2 \label{nearest_neighbors}\end{equation}

* Perform principal component analysis of each local neighborhood patch.
  For each \( x \in X \) compute the Gram matrix \( G \) of its 
  neighbors such that \( G_{i,j} = (\mathcal{N}_x^{i} , \mathcal{N}_x^{j}) \) and center 
  it. Compute its \( t \) (the number of required features) eigenvectors and 
  store it in the matrix \( V \). Compute matrix 
  \[ Q = \begin{bmatrix} 1_k & V \end{bmatrix} \begin{bmatrix} 1_k \\ V \end{bmatrix} \]
  and put it to the sparse alignment matrix \( L \) (initially set by zeroes) using the following procedure:
  \[ L \leftarrow L + E_k - Q. \]

* Embedding through eigendecomposition. To obtain \( t \) features (coordinates) 
  of embedded vectors solve the partial eigenproblem 
  \[ L f = \lambda f, \]
  for smallest eigenvalues \( \lambda_1, \dots, \lambda_t, \lambda_{t+1} \) and 
  its corresponding eigenvectors \( f_1, \dots, f_t, f_{t+1} \). Drop the smallest eigenvalue
  \( \lambda_1 \sim 0 \) (with the corresponding eigenvector) and form embedding matrix 
  such that \(i\)-th coordinate (\( i=1,\dots,N \)) of \(j\)-th eigenvector 
  (\( j=1,\dots,t\) ) corresponds to \(j\)-th coordinate of projected \(i\)-th vector.

Kernel Local Tangent Space Alignment
------------------------------------

Like the Locally Linear Embedding algorithm, LTSA allows generalization for Mercer kernel functions. 
Nearest neighbors computation in KLTSA is identical to one in KLTSA and the 
matrix \( G \) in the step 2 is naturally replaced with matrix \( K_{i,j} = k(\mathcal{N}_x^{i},\mathcal{N}_x^{j}) \).

References
----------

* Zhang, Z., & Zha, H. (2002). 
  [Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment](http://arxiv.org/abs/cs/0212008)

