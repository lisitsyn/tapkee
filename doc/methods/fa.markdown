Factor analysis
---------------

Factor analysis aims at describing how several observed variables are correlated
to each other by means of identifying a set of unobserved variables, the
so-called factors. Desirably, the number of factors is shorter than the number
of observed variables.

Factor analysis is an iterative algorithm. First of all the projection matrix
is initialized randomly and the factors variance is set to the identity. Then,
every iteration consists of the following steps:

* Compute the regularized inverse covariance matrix of the projection.

* Update the factors variance matrix.

* Update the projection matrix.

* Check for convergence using the log-likelihood of the model. If the difference
between the current log-likelihood and the previous iteration's log-likelihood
is below a threshold, then the algorithm has converged.

References
----------

* Spearman, C. (1904). [General Intelligence, Objectively Determined and
Measured](http://www.mendeley.com/catalog/general-intelligence-objectively-determined-measured/).
