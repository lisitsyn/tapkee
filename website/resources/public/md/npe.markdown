Neighborhood Preserving Embedding
---------------------------------

The Neighborhood Preserving Embedding (NPE) algorithm can be considered 
as a linear approximation of the Locally Linear Embedding algorithm. 
Thus most of computation routines can be shared with LLE. The NPE algorithm
uses steps 1 and 2 of the Locally Linear Embedding and the main 
difference lies in the eigendecomposition based embedding.

According to the NPE algorithm embedding can be found from the solution of 
the following partial generalized eigenproblem:
$$ R L R f = \lambda R R^T f $$
where $ R $ is a matrix containing all feature 
vectors $ x\_1 , \dots , x\_N $ row-wise. The problem is solved for 
smallest eigenvalues $ \lambda\_1, \dots, \lambda\_t $ and 
its corresponding eigenvectors $ f\_1, \dots, f\_t $. The final embedding 
is obtained with a matrix such that $i$-th coordinate ($ i=1,\dots,N $) 
of $j$-th eigenvector ($ j=1,\dots,t$ ) corresponds to $j$-th 
coordinate of projected $i$-th vector.

References

* He, X., Cai, D., Yan, S., & Zhang, H.-J. (2005). 
  [Neighborhood preserving embedding](https://ieeexplore.ieee.org/document/1544858)

