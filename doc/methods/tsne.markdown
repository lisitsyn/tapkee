Stochastic Neighbour Embedding
------------------------------

Stochastic Neighbour Embedding (SNE) uses conditional probability densities in order
to model pairwise similarities between data points, rather than using Euclidean distances
directly. The similarity of the point $x\_j$ to the point $x\_i$ is the conditional probability
$p\_{j|i}$, which is the probability that $x\_j$ would be $x\_i$'s neighbour taking into account that
neighbourhoods are built in proportion to Gaussian probability densities centered
at $x\_{i}$. Formally,
$$ p\_{j|i} = \frac{exp(-\|x\_i - x\_j\|^2 / 2 \sigma\_i^2)}{\sum\_{k \neq i} exp(-\|x\_i - x\_k\|^2 / 2 \sigma\_i^2)} $$
where $\sigma\_i$ is the variance of the Gaussian centered on $x\_i$, whose
computation will be later explained. The similarities in the
low-dimensional space are defined in a similar way. However, the variance of
the Gaussian distributions employed are fixed to $\frac{1}{\sqrt{2}}$ this time, i.e.
$$ q\_{j|i} = \frac{exp(-\|x\_i - x\_j\|^2)}{\sum\_{k \neq i} exp(-\|x\_i - x\_k\|^2)}. $$

Intuitively, if the low-dimensional points $y\_i$ and $y\_j$ map correctly the similarity
between their high-dimensional counterparts $x\_i$ and $x\_j$, then $p\_{j|i}$ and $q\_{j|i}$
will be close to each other. SNE aims at making these quantities as close as possible
minimizing the sum of Kullback-Leibler divergences over all the data set. Thus, the cost
function is
$$C = \sum\_{i} KL(P\_i \| Q\_i) = \sum\_{i} \sum\_{j} p\_{j|i} \log \frac{p\_{j|i}}{q\_{j|i}}.$$

Unfortunately, there exists no optimal value of the Gaussian variance for all the points in
the data set since the data may vary considerably throughout the data set.
SNE chooses the value of each $\sigma\_i$ performing binary search so that a user specified
value for the Shannon entropy of $P\_i$ is achieved.

t-Distributed Stochastic Neighbour Embedding
--------------------------------------------

There are two main issues related to SNE that t-Distributed Stochastic Neighbour Embedding
(t-SNE) addresses:

* SNE's cost function using gradient descent is faster optimized
using symmetric similarities. Therefore, t-SNE uses joint probability distributions $p\_{ij}$
and $q\_{ij}$ instead of conditional distributions.

* t-SNE uses Student's t instead of Gaussian distributions to handle better the so-called
crowding problem.

References
----------

* Van der Maaten, L., Hinton, G. (2008). [Visualizing Data using t-SNE](http://jmlr.csail.mit.edu/papers/v9/vandermaaten08a.html).
