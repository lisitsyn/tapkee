Locally Linear Embedding
------------------------

Given a set of feature vectors \(X = \{ x_1, x_2, \dots x_N \} \) the Locally Linear Embedding algorithm 
proposes to perform the following steps:

* Identify nearest neighbors. For each \( x \in X \) identify its \( k \) nearest 
  neighbors, i.e. a set \( \mathcal{N}_x \) of \( k \) feature vectors such that 
  \begin{equation} \arg\min_{\mathcal{N}_x} \sum_{x_n \in \mathcal{N}_x}\| x - x_n \|_2^2 \label{nearest_neighbors} \end{equation}

* Compute linear reconstruction weights. For each \( x \in X \) compute weight vector 
  \( w \in \mathbb{R}^n \) that minimizes 
  \[ \| x - \sum_{i=1}^{k} w_i \mathcal{N}_x^{i} \|_2, ~~ \text{w.r.t.} ~~ \|w\|_2 = 1\]
  where \( \mathcal{N}_x^{i} \) is a \( i\)-th element of the set \( \mathcal{N}_x \). 
  The solution of the problem stated above can be found from the normalized solution of the following equation:
  \begin{equation} G w = 1_k, \label{weights}\end{equation}
  where \( G \) is a \( k \times k \) matrix such 
  that \( G_{i,j} = (x - \mathcal{N}_x^{i})(x - \mathcal{N}_x^{j}) \) 
  and \( 1_k \in \mathbb{R}^k \) is a vector of all ones. Obviously, the 
  problem comes ill-posed in case \( k \) gets more than dimension of 
  feature space \( X \). This can be avoided with the regularization:
  \[ G \leftarrow G + \varepsilon E, \]
  where \(E\) is an identity matrix and \( \varepsilon \) is a pre-defined 
  constant reconstruction shift (usually \( 10^{-3}\)).
  Once \( w \) is computed it is stored into the sparse alignment matrix 
  \( L \) (initially set by zero) with the following procedure:
  \[ L_{I,I} \leftarrow L_{I,I} + W, \]
  where \( I \) is a set containing indices of all element 
  of the set \(\mathcal{N}_x\) and \( x \) itself, \( L_{I,I} \) denotes all 
  \( (i,j) \) elements of the sparse matrix L such that 
  \( i,j \in I\) and \[ W = \begin{bmatrix} 1 & -w \\ -w^{T} & w^T w \end{bmatrix}\].

* Embedding through eigendecomposition. To obtain \( t \) features (coordinates) of 
  embedded vectors solve the partial eigenproblem 
  \[ L f = \lambda f, \]
  for smallest eigenvalues \( \lambda_1, \dots, \lambda_t, \lambda_{t+1} \) and 
  its corresponding eigenvectors \( f_1, \dots, f_t, f_{t+1} \). Drop the smallest eigenvalue
  \( \lambda_1 \sim 0 \) (with the corresponding eigenvector) and form embedding matrix 
  such that \(i\)-th coordinate (\( i=1,\dots,N \)) of \(j\)-th eigenvector 
  (\( j=1,\dots,t\) ) corresponds to \(j\)-th coordinate 
  of projected \(i\)-th vector.

Kernel Locally Linear Embedding
-------------------------------

The Locally Linear Embedding algorithm can be generalized for spaces with 
defined dot product function \( k(x,y) \) (so-called [RKHS](http://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space))
in the elegant way. Using the following equation
\[ || x - y||_2^2 = (x,x) - 2 (x,y) + (y,y) \]
we may transform \eqref{nearest_neighbors} to the following form:
\[ \arg\min_{\mathcal{N}_x} \sum_{x_n \in \mathcal{N}_x} \left[ k(x,x) - 2 k(x,x_n) + k(x_n, x_n) \right] . \]
The matrix \( G \) in the equation \eqref{weights} can be formulated in terms 
of dot product as well. To find \( G \) using only dot products we can compute 
the Gram matrix \( K \) such that \( K_{i,j} = k(x_i, x_j)\) and center it 
using the matrix \( C_k = E_k - \frac{1}{k} 1 1^{T}\):
\[ G = K C_k K. \]
There is an efficient way to compute that - it is can be done with
subtracting a column mean of \( K \) from each column of \( K \), 
subtracting a row mean of \( K \) from each row of \( K \) and 
adding the grand mean of all elements of \( K \) to \( K \).

References
----------

* [Sam Roweis' page on LLE](http://www.cs.nyu.edu/~roweis/lle/)
          <li><a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.123.7319&rep=rep1&type=pdf">Saul, L. K., Ave, P., Park, F., & Roweis, S. T. (2001). An introduction to Locally Linear Embedding</a></li>
          <li><a href="http://linkinghub.elsevier.com/retrieve/pii/S0031320306002160">Zhao, D. (2006) Formulating LLE using alignment technique</a></li>
        </ol>
