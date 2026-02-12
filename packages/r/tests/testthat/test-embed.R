test_that("PCA works on random data", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "pca", target_dimension = 2L)
  expect_true(is.matrix(result))
  expect_equal(nrow(result), 100)
  expect_equal(ncol(result), 2)
})

test_that("Isomap works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "isomap",
                         num_neighbors = 15L, target_dimension = 2L)
  expect_equal(dim(result), c(100, 2))
})

test_that("LLE works with defaults", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X)
  expect_equal(dim(result), c(100, 2))
})

test_that("t-SNE works", {
  set.seed(42)
  X <- matrix(rnorm(500), nrow = 5, ncol = 100)
  result <- tapkee_embed(X, method = "t-sne",
                         target_dimension = 2L, sne_perplexity = 10.0)
  expect_equal(dim(result), c(100, 2))
})

test_that("MDS works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "mds", target_dimension = 2L)
  expect_equal(dim(result), c(100, 2))
})

test_that("Diffusion Map works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "dm", target_dimension = 2L,
                         gaussian_kernel_width = 1.0)
  expect_equal(dim(result), c(100, 2))
})

test_that("Factor Analysis works", {
  set.seed(42)
  X <- matrix(rnorm(500), nrow = 5, ncol = 100)
  result <- tapkee_embed(X, method = "fa", target_dimension = 2L,
                         max_iteration = 50L)
  expect_equal(dim(result), c(100, 2))
})

test_that("neighbors_method brute works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "isomap",
                         neighbors_method = "brute",
                         num_neighbors = 15L, target_dimension = 2L)
  expect_equal(dim(result), c(100, 2))
})

test_that("neighbors_method vptree works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "isomap",
                         neighbors_method = "vptree",
                         num_neighbors = 15L, target_dimension = 2L)
  expect_equal(dim(result), c(100, 2))
})

test_that("eigen_method dense works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "pca",
                         eigen_method = "dense",
                         target_dimension = 2L)
  expect_equal(dim(result), c(100, 2))
})

test_that("eigen_method randomized works", {
  set.seed(42)
  X <- matrix(rnorm(300), nrow = 3, ncol = 100)
  result <- tapkee_embed(X, method = "pca",
                         eigen_method = "randomized",
                         target_dimension = 2L)
  expect_equal(dim(result), c(100, 2))
})

test_that("unknown neighbors_method raises error", {
  X <- matrix(rnorm(30), nrow = 3, ncol = 10)
  expect_error(tapkee_embed(X, method = "isomap",
                            neighbors_method = "nonexistent"),
               "Unknown neighbors method")
})

test_that("unknown eigen_method raises error", {
  X <- matrix(rnorm(30), nrow = 3, ncol = 10)
  expect_error(tapkee_embed(X, method = "pca",
                            eigen_method = "nonexistent"),
               "Unknown eigen method")
})

test_that("unknown method raises error", {
  X <- matrix(rnorm(30), nrow = 3, ncol = 10)
  expect_error(tapkee_embed(X, method = "nonexistent"), "Unknown method")
})

test_that("non-matrix input raises error", {
  expect_error(tapkee_embed(1:10, method = "pca"), "must be a numeric matrix")
})

test_that("target_dimension = 3 works", {
  set.seed(42)
  X <- matrix(rnorm(500), nrow = 5, ncol = 100)
  result <- tapkee_embed(X, method = "pca", target_dimension = 3L)
  expect_equal(dim(result), c(100, 3))
})
