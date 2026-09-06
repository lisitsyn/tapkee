library(tapkee)

N <- 100

# Create 1D data: integer values from 0 to N-1
# The pairwise distance d(i,j) = |i - j| is computed internally
data <- matrix(as.numeric(0:(N - 1)), nrow = 1)

# Embed using Multidimensional Scaling to 1D
result <- tapkee_embed(data, method = "mds", target_dimension = 1L)

print(result)
