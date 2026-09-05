import numpy as np
import tapkee

# Simple 1D data: values from 0 to 99
data = np.arange(100, dtype=np.float64).reshape(1, -1)

# Embed using Multidimensional Scaling to 1D
embedding = tapkee.embed(data, method='mds', target_dimension=1)

print(embedding)
