Diffusion map
-------------

The diffusion map algorithm performs the following steps to embed 
feature vectors $ x\_1, \dots, x\_N $:
  
* Compute $ N \times N $ gaussian kernel matrix $ K $ such that 
  $$ K\_{i,j} = exp \left\\{ - \frac{d^2(x\_i,x\_j)}{\omega} \right\\},$$
  where $ d : X \times X \to \mathbb{R} $ is a distance function and
  $ \omega > 0 $ is a width of the kernel.
 
* Transform the matrix $ K $ using the following equations
  $$ K\_{i,j} \leftarrow \frac{K\_{i,j}}{(p\_i p\_j)^q},$$ where $ p\_i = \sum\_{j=1}^{N} K\_{j,i}$. 
  Only $ q=1 $ for 'standard' diffusion map is currently supported.
  Then, recompute $ p\_i = \sum\_{j=1}^{N} K\_{j,i} $ again and do
  $$ K\_{i,j} \leftarrow \frac{K\_{i,j}}{\sqrt{p\_i p\_j}}.$$
  
* Construct embedding with $ \dim = d $ from the solution of the following partial eigenproblem
  $$ K f = \lambda f $$
  for $d+1 $ largest eigenvalues. Form the embedding matrix such that the 
  $i$-th coordinate ($ i=1,\dots,N $) of $j$-th largest
  eigenvector ($ j=2,\dots,d+1$ ) corresponds to $j$-th coordinate 
  of projected $i$-th vector, normalized by $\lambda_i^t$ and the first eigenvector 
  corresponding to $\lambda_1 = 1$.

References
----------

* Coifman, R., & Lafon, S. (2006). 
  [Diffusion maps](http://linkinghub.elsevier.com/retrieve/pii/S1063520306000546)

