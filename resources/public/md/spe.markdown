Stochastic Proximity Embedding
------------------------------

Stochastic Proximity Embedding (SPE) .  given a set 
of feature vectors $X = \{ x\_1, x\_2, \dots x\_N \} $ and/or their symmetric distance matrix $ D\_{ij} $.
It is an interesting method because of its simplicity and efficiency, it scales linearly with the sample size $ N $.

SPE consists of the following steps:

* Choose an initial learning rate $ \lambda $. This parameter will decrease later 
  in order to avoid oscillatory behaviour.

* Initialize randomly the point coordinates in the embedded 
  space $Y = \{ y\_1, y\_2, \dots y\_N \} $.

* Update a set of the point coordinates. For a prescribed number of iterations $ S $, 
  choose a pair of point indices $ i $ and $ j $ and compute their distances in the 
  embedded space, 
  $$ d\_{i,j} = \| x\_i - x\_j \| $$ 
  If $ d\_{i,j} \neq D\_{i,j} $ then update the coordinates of the selected points by:
  $$ x\_i \leftarrow x\_i + \lambda \frac{1}{2} \frac{D\_{ij} - d\_{ij}}{d\_{ij} + \epsilon} (x\_i - x\_j), $$
  $$ x\_j \leftarrow x\_j + \lambda \frac{1}{2} \frac{D\_{ij} - d\_{ij}}{d\_{ij} + \epsilon} (x\_j - x\_i). $$

* Decrease the learning rate $ \lambda $ by a quantity $ \delta \lambda $.

* Repeat steps 2 to 4 for a predetermined number of iterations $ C $.

References
----------

* Agrafiotis, D. K. (2002). 
  [Stochastic Proximity Embedding](http://www.dimitris-agrafiotis.com/Papers/jcc20078.pdf)
