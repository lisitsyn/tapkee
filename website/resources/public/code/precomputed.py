import numpy as np
import tapkee

N = 100

# Create 1D data: integer values from 0 to N-1
# The pairwise distance d(i,j) = |i - j| is computed internally
data = np.arange(N, dtype=np.float64).reshape(1, -1)

# Embed using Multidimensional Scaling to 1D
embedding = tapkee.embed(data, method='mds', target_dimension=1)

print(embedding)
