Manifold Sculpting
------------------

Manifold Sculpting is a nonlinear dimensionality reduction algorithm that
iteratively reduces dimensionality by dropping one dimension at a time while
preserving the local structure of the data. Given a set of feature vectors
$X = \{ x\_1, x\_2, \dots x\_N \} $ the algorithm performs the following steps:

* Identify nearest neighbors. For each $ x \in X $ find its $ k $ nearest
  neighbors and record the distances and relative angles to those neighbors.

* Compute a scaling factor $ \sigma \in (0,1) $ that controls the rate of
  convergence. Typical values are close to $ 0.99 $.

* Iteratively sculpt the manifold. At each iteration:

  * Scale the data in the dimension to be dropped by $ \sigma $,
    gradually collapsing that dimension toward zero.

  * Adjust the positions of all points to restore the original
    neighbor relationships. Each point $ x\_i $ is moved to minimize the
    cost function
    $$ E = \sum\_{i=1}^{N} \sum\_{x\_j \in \mathcal{N}\_{x\_i}} \left( \frac{\| x\_i - x\_j \| - D\_{ij}}{D\_{ij}} \right)^2 $$
    where $ D\_{ij} $ is the original distance between $ x\_i $ and its neighbor
    $ x\_j $, and $ \mathcal{N}\_{x\_i} $ is the set of $ k $ nearest neighbors
    of $ x\_i $.

  * Heuristically adjust points to also preserve the angles between
    neighboring vectors.

* Repeat the scaling and adjustment steps until convergence, then drop the
  collapsed dimension and proceed to the next one. Continue until the desired
  target dimensionality $ t $ is reached.

The algorithm can be seen as a form of force-directed optimization that
gradually squeezes out dimensions while trying to preserve the local geometry
of the data manifold. Unlike spectral methods, it does not require solving
eigenproblems, making it conceptually simple and straightforward to implement.

References
----------

* Gashler, M. S., Ventura, D., & Martinez, T. (2007).
  [Manifold Sculpting](https://axon.cs.byu.edu/papers/gashler2007icml.pdf)
