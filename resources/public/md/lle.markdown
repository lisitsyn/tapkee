Locally Linear Embedding
------------------------

Given a set of feature vectors $X = \{ x\_1, x\_2, \dots x\_N \} $ the Locally Linear Embedding algorithm 
proposes to perform the following steps:

* Identify nearest neighbors. For each $ x \in X $ identify its $ k $ nearest 
  neighbors, i.e. a set $ \mathcal{N}\_x $ of $ k $ feature vectors such that 
  $$ \arg\min\_{\mathcal{N}\_x} \sum\_{x\_n \in \mathcal{N}\_x}\| x - x\_n \|\_2^2 $$

* Compute linear reconstruction weights. For each $ x \in X $ compute weight vector 
  $ w \in \mathbb{R}^n $ that minimizes 
  $$ \| x - \sum\_{i=1}^{k} w\_i \mathcal{N}\_x^{i} \|\_2, ~~ \text{w.r.t.} ~~ \|w\|\_2 = 1$$
  where $ \mathcal{N}\_x^{i} $ is a $ i$-th element of the set $ \mathcal{N}\_x $. 
  The solution of the problem stated above can be found from the normalized solution of the following equation:
  $$ G w = 1\_k, $$
  where $ G $ is a $ k \times k $ matrix such 
  that $ G\_{i,j} = (x - \mathcal{N}\_x^{i})(x - \mathcal{N}\_x^{j}) $ 
  and $ 1\_k \in \mathbb{R}^k $ is a vector of all ones. Obviously, the 
  problem comes ill-posed in case $ k $ gets more than dimension of 
  feature space $ X $. This can be avoided with the regularization:
  $$ G \leftarrow G + \varepsilon E, $$
  where $E$ is an identity matrix and $ \varepsilon $ is a pre-defined 
  constant reconstruction shift (usually $ 10^{-3}$).
  Once $ w $ is computed it is stored into the sparse alignment matrix 
  $ L $ (initially set by zero) with the following procedure:
  $$ L\_{I,I} \leftarrow L\_{I,I} + W, $$
  where $ I $ is a set containing indices of all element 
  of the set $\mathcal{N}\_x$ and $ x $ itself, $ L\_{I,I} $ denotes all 
  $ (i,j) $ elements of the sparse matrix L such that 
  $ i,j \in I$ and $$ W = \begin{bmatrix} 1 & -w \\ -w^{T} & w^T w \end{bmatrix}$$.

* Embedding through eigendecomposition. To obtain $ t $ features (coordinates) of 
  embedded vectors solve the partial eigenproblem 
  $$ L f = \lambda f, $$
  for smallest eigenvalues $ \lambda\_1, \dots, \lambda\_t, \lambda\_{t+1} $ and 
  its corresponding eigenvectors $ f\_1, \dots, f\_t, f\_{t+1} $. Drop the smallest eigenvalue
  $ \lambda\_1 \sim 0 $ (with the corresponding eigenvector) and form embedding matrix 
  such that $i$-th coordinate ($ i=1,\dots,N $) of $j$-th eigenvector 
  ($ j=1,\dots,t$ ) corresponds to $j$-th coordinate 
  of projected $i$-th vector.

Kernel Locally Linear Embedding
-------------------------------

The Locally Linear Embedding algorithm can be generalized for spaces with 
defined dot product function $ k(x,y) $ (so-called [RKHS](http://en.wikipedia.org/wiki/Reproducing\_kernel\_Hilbert\_space))
in the elegant way. Using the following equation
$$ || x - y||\_2^2 = (x,x) - 2 (x,y) + (y,y) $$
we may transform the nearest neighbors problem to the following form:
$$ \arg\min\_{\mathcal{N}\_x} \sum\_{x\_n \in \mathcal{N}\_x} \left[ k(x,x) - 2 k(x,x\_n) + k(x\_n, x\_n) \right] . $$
The matrix $ G $ can be formulated in terms 
of dot product as well. To find $ G $ using only dot products we can compute 
the Gram matrix $ K $ such that $ K\_{i,j} = k(x\_i, x\_j)$ and center it 
using the matrix $ C\_k = E\_k - \frac{1}{k} 1 1^{T}$:
$$ G = K C\_k K. $$
There is an efficient way to compute that - it is can be done with
subtracting a column mean of $ K $ from each column of $ K $, 
subtracting a row mean of $ K $ from each row of $ K $ and 
adding the grand mean of all elements of $ K $ to $ K $.

References
----------

* [Sam Roweis' page on LLE](https://web.archive.org/web/2023/https://cs.nyu.edu/~roweis/lle/)
* Saul, L. K., Ave, P., Park, F., & Roweis, S. T. (2001).
  [An introduction to Locally Linear Embedding](https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf)
* Zhao, D. (2006) [Formulating LLE using alignment technique](http://linkinghub.elsevier.com/retrieve/pii/S0031320306002160)
