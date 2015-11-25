Stochastic Proximity Embedding
------------------------------

Stochastic Proximity Embedding (SPE) acts on a set of $ N $ vectors $Y = \{ y\_1, y\_2, \dots y\_N \} $ with corresponding symmetric distance matrix $ D\_{ij} $ in the following manner:

1. Choose an initial learning rate $ \lambda$.

2. Initialize randomly the point coordinates in the embedded space $X = \{ x\_1, x\_2, \dots x\_N \} $.

3. Select at random a pair of points with indices $ i $ and $ j $. For a prescribed number of iterations $ S $, 
  compute their distances in the embedded space, 
  $$ d\_{i,j} = \| x\_i - x\_j \| $$; 
  if $ d\_{i,j} \neq D\_{i,j} $ then update the coordinates of the selected points by
  $$ x\_i \leftarrow x\_i + \lambda \frac{1}{2} \frac{D\_{ij} - d\_{ij}}{d\_{ij} + \epsilon} (x\_i - x\_j), $$
  $$ x\_j \leftarrow x\_j + \lambda \frac{1}{2} \frac{D\_{ij} - d\_{ij}}{d\_{ij} + \epsilon} (x\_j - x\_i). $$

4. Decrease the learning rate $ \lambda $ by  $ \delta \lambda | 0 < \delta < 1 $. $ \lambda $ is decreased to avoid oscillatory behaviour.

5. Repeat steps 3 and 4 for a predetermined number of iterations $ C $.

SPE is an interesting method because of its simplicity and efficiency, as it scales linearly with the sample size $ N $.

Reference
---------

* D. K. Agrafiotis. "Stochastic Proximity Embedding," *Journal of Computational Chemistry*, 2003.
