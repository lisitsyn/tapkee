In this example RNA sequencies are embedded using the match kernel 
`MatchKernelCallback` (that is essentially an inverse of a Hamming distance). 
RNA sequencies are supposed to be read from the file provided via a command line 
argument (one sequence per line). As the Kernel Locally Linear Embedding algorithm 
requires only a kernel callback we just pass only the kernel callback using
the `withKernel` member function. 
