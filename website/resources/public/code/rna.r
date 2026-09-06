library(tapkee)

# Read RNA sequences from file
args <- commandArgs(trailingOnly = TRUE)
lines <- readLines(args[1])
sequences <- lines[nchar(lines) > 0]

# One-hot encode: each base becomes a 4-element binary vector
base_index <- c(A = 1, C = 2, G = 3, U = 4, T = 4)
seq_len <- nchar(sequences[1])
n <- length(sequences)
data <- matrix(0, nrow = 4 * seq_len, ncol = n)
for (j in seq_along(sequences)) {
  chars <- strsplit(sequences[j], "")[[1]]
  for (i in seq_along(chars)) {
    data[4 * (i - 1) + base_index[chars[i]], j] <- 1
  }
}

# Embed using Locally Linear Embedding
result <- tapkee_embed(data, method = "lle", num_neighbors = 30L)

print(result)
