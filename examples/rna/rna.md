In this example RNA sequencies are embedded using the match kernel 
`match_kernel_callback` (that is essentially an inverse of a Hamming distance). 
RNA sequencies are supposed to be read from the file provided via a command line 
argument (one sequence per line). As the Kernel Locally Linear Embedding algorithm 
requires only a kernel callback - dummy callbacks for distance computation 
and feature vector access are used.
