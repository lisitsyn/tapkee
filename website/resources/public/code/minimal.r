library(tapkee)

# Simple 1D data: values from 0 to 99
data <- matrix(as.numeric(0:99), nrow = 1)

# Embed using Multidimensional Scaling to 1D
result <- tapkee_embed(data, method = "mds", target_dimension = 1L)

print(result)
