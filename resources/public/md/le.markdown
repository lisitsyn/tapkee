Laplacian Eigenmaps
-------------------

The Laplacian Eigenmaps algorithm performs the following simple steps 
to embed given feature vectors $ x\_1, \dots, x\_N $:

* Identify nearest neighbors. For each $ x \in X $ identify its $ k $ nearest neighbors, 
  i.e. a set $ \mathcal{N}\_x $ of $ k $ feature vectors such that
  $$ \arg\min\_{\mathcal{N}\_x} \sum\_{x\_n \in \mathcal{N}\_x} d(x,x\_n), $$
  where $ d : X \times X \to \mathbb{R} $ is a distance function.

* Construct weight matrix. Initially setting $N\times N$ matrix $ W $ to zero, set
  $$ W\_{i,j} = exp \left\\{ - \frac{d^2(x\_i,x\_j)}{\tau} \right\\} $$
  iff for $i$-th vector $x\_i$ neighbors set $\mathcal{N}\_{x\_i}$ contains $ x\_j $ and 
  vice versa (so-called mutual neighborhood). Find a diagonal matrix $ D $ such that 
  $ D\_{i,i} = \sum\_{j=1}^{N} W\_{j,i} $.

* Find embedding throught eigendecomposition. To obtain $ t $ features (coordinates) of embedded 
  vectors solve the partial generalized eigenproblem 
  $$ (D-W) f = \lambda D f, $$
  for smallest eigenvalues $ \lambda\_1, \dots, \lambda\_t, \lambda\_{t+1} $ and its corresponding 
  eigenvectors $ f\_1, \dots, f\_t, f\_{t+1} $. Drop the smallest eigenvalue
  $ \lambda\_1 \sim 0 $ (with the corresponding eigenvector) and form embedding 
  matrix such that $i$-th coordinate ($ i=1,\dots,N $) of $j$-th eigenvector 
  ($ j=1,\dots,t$ ) corresponds to $j$-th coordinate of projected $i$-th vector.

References
----------

* Belkin, M., & Niyogi, P. (2002). 
  [Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.9400&rep=rep1&type=pdf)

