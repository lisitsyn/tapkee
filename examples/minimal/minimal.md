In this example the simplest case of using the Tapkee library is considered. For
the sake of simplicity, the input data used in this example is a one dimensional range
of real values from 0.0 to 99.0 with step 1.0. Therefore, it actually does not reduce 
the dimensionality but maps vectors using the provided distances. In this example 
the Multidimensional Scaling (MDS) algorithm is used (in its classic metric formulation).
MDS requires only distance callback and the `MyDistanceCallback` struct implements 
that with the absolute value of the difference between two data points. The output of 
such preprocessing is the centered input data (i.e. 0.0 maps to -49.5 and 99.0 maps to 49.5) 
