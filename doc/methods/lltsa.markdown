Linear Local Tangent Space Alignment
------------------------------------

The Linear Local Tangent Space Alignment is a modification of the LTSA algorithm. 
Main difference (just like in NPE and LLE) of linear and original LTSA methods lies 
in the way of constructing embedding. Instead of solving common for LLE and LTSA 
eigenproblem, LLTSA requires solving the following generalized eigenproblem:
$$ R L R^T f = \lambda R R^T f, $$
where $ R $ is a matrix containing all feature vectors $ x\_1, \dots, x\_N $ row-wise. 
The problem is solved for smallest eigenvalues $ \lambda\_1 \leq \lambda\_2 \leq \dots \leq \lambda\_t $ 
and its corresponding eigenvectors $ f\_1, \dots, f\_t $. To find final embedding LLTSA forms a
matrix such that $i$-th coordinate ($ i=1,\dots,N $) of $j$-th eigenvector ($ j=1,\dots,t$ ) 
corresponds to $j$-th coordinate of projected $i$-th vector.

References
----------

* Zhang, T., Yang, J., Zhao, D., & Ge, X. (2007). 
  [Linear local tangent space alignment and application to face recognition](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.85.2698)

