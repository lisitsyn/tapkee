Random projection
-----------------

The Random projection algorithm is yet more simple algorithm (comparing to PCA and MDS). 
It can be said that the algorithm is based on Johnson-Lindenstrauss lemma that states 
that a small number of vectors in high-dimensional space can be embedded into a 
space of much lower dimension with keeping pairwise distances nearly preserved. 
The algorithm itself is: 

* Construct random basis matrix $ P $ with normalized random gaussian vectors as columns.
* Project data with left multiplication with generated matrix $ Y = P X $.
