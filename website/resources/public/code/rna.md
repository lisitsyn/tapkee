In this example RNA sequences are embedded using dimensionality reduction.
RNA sequences are read from a file (one sequence per line). The C++ version
uses a custom match kernel callback (essentially an inverse Hamming distance)
with Kernel Locally Linear Embedding. The Python and R versions one-hot
encode the sequences and use Locally Linear Embedding.
