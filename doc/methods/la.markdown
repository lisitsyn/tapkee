Laplacian Eigenmaps
-------------------

The Laplacian Eigenmaps algorithm performs the following simple steps 
to embed given feature vectors \( x_1, \dots, x_N \):

* Identify nearest neighbors. For each \( x \in X \) identify its \( k \) nearest neighbors, 
  i.e. a set \( \mathcal{N}_x \) of \( k \) feature vectors such that
  \[ \arg\min_{\mathcal{N}_x} \sum_{x_n \in \mathcal{N}_x} d(x,x_n), \]
  where \( d : X \times X \to \mathbb{R} \) is a distance function.

* Construct weight matrix. Initially setting \(N\times N\) matrix \( W \) to zero, set
  \begin{equation} W_{i,j} = \exp \left\{ - \frac{d^2(x_i,x_j)}{\tau} \right\} \label{weight}\end{equation}
  iff for \(i\)-th vector \(x_i\) neighbors set \(\mathcal{N}_{x_i}\) contains \( x_j \) and 
  vice versa (so-called mutual neighborhood). Find a diagonal matrix \( D \) such that 
  \( D_{i,i} = \sum_{j=1}^{N} W_{j,i} \).

* Find embedding throught eigendecomposition. To obtain \( t \) features (coordinates) of embedded 
  vectors solve the partial generalized eigenproblem 
  \[ (D-W) f = \lambda D f, \]
  for smallest eigenvalues \( \lambda_1, \dots, \lambda_t, \lambda_{t+1} \) and its corresponding 
  eigenvectors \( f_1, \dots, f_t, f_{t+1} \). Drop the smallest eigenvalue
  \( \lambda_1 \sim 0 \) (with the corresponding eigenvector) and form embedding 
  matrix such that \(i\)-th coordinate (\( i=1,\dots,N \)) of \(j\)-th eigenvector 
  (\( j=1,\dots,t\) ) corresponds to \(j\)-th coordinate of projected \(i\)-th vector.

References
----------

* Belkin, M., & Niyogi, P. (2002). 
  [Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.9400&rep=rep1&type=pdf)

