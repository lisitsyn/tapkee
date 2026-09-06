import sys
import numpy as np
import tapkee

# Read RNA sequences from file (one sequence per line)
with open(sys.argv[1]) as f:
    sequences = [line.strip() for line in f if line.strip()]

# One-hot encode: each base becomes a 4-element binary vector
base_index = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}
seq_len = len(sequences[0])
n = len(sequences)
data = np.zeros((4 * seq_len, n))
for j, seq in enumerate(sequences):
    for i, base in enumerate(seq):
        data[4 * i + base_index[base], j] = 1.0

# Embed using Locally Linear Embedding
embedding = tapkee.embed(data, method='lle', num_neighbors=30)

print(embedding)
