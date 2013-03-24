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
  Then, recompute $ p\_i = \sum\_{j=1}^{N} K\_{j,i} $ again and do
  $$ K\_{i,j} \leftarrow \frac{K\_{i,j}}{\sqrt{p\_i p\_j}}.$$
  
* Construct embedding with $ \dim = t $ from the solution of the following partial eigenproblem
  $$ K^{T} K f = \lambda f $$
  for $t $ largest eigenvalues. Form the embedding matrix such that 
  $i$-th coordinate ($ i=1,\dots,N $) of $j$-th largest
  eigenvector ($ j=1,\dots,t$ ) corresponds to $j$-th coordinate 
  of projected $i$-th vector.

References
----------

* Coifman, R., & Lafon, S. (2006). 
  [Diffusion maps](http://linkinghub.elsevier.com/retrieve/pii/S1063520306000546)

