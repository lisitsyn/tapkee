In this example a precomputed distance structure is used for embedding with
Multidimensional Scaling (MDS). The distances are absolute values of differences
between indices. The C++ version demonstrates the precomputed distance callback
API, while the Python and R versions pass the data directly and let Tapkee
compute distances internally. The output is a range of real values from
-49.5 to 49.5.
