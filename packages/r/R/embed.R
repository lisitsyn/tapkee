#' Dimensionality Reduction with Tapkee
#'
#' Embed high-dimensional data into a lower-dimensional space using
#' one of 20+ algorithms from the Tapkee C++ library.
#'
#' @param data A numeric matrix where rows are features and columns are
#'   samples. Each column is one data point.
#' @param method Character string specifying the reduction method. Short names:
#'   \code{"lle"}, \code{"isomap"}, \code{"pca"}, \code{"t-sne"},
#'   \code{"mds"}, \code{"dm"}, \code{"kpca"}, \code{"la"}, \code{"lpp"},
#'   \code{"npe"}, \code{"ltsa"}, \code{"lltsa"}, \code{"hlle"},
#'   \code{"l-mds"}, \code{"l-isomap"}, \code{"spe"}, \code{"fa"},
#'   \code{"random_projection"}, \code{"manifold_sculpting"}, \code{"passthru"}.
#' @param num_neighbors Integer. Number of nearest neighbors for local methods.
#'   Default in Tapkee: 5.
#' @param target_dimension Integer. Output dimensionality.
#'   Default in Tapkee: 2.
#' @param gaussian_kernel_width Numeric. Kernel width for Laplacian Eigenmaps,
#'   Diffusion Maps, LPP. Default in Tapkee: 1.0.
#' @param landmark_ratio Numeric in \eqn{[0, 1]}. Ratio of landmarks for
#'   Landmark Isomap and Landmark MDS. Default in Tapkee: 0.5.
#' @param max_iteration Integer. Maximum iterations for iterative methods.
#'   Default in Tapkee: 100.
#' @param diffusion_map_timesteps Integer. Number of timesteps for Diffusion
#'   Maps. Default in Tapkee: 3.
#' @param sne_perplexity Numeric. Perplexity for t-SNE.
#'   Default in Tapkee: 30.0.
#' @param sne_theta Numeric. Barnes-Hut theta for t-SNE approximation.
#'   Default in Tapkee: 0.5.
#' @param squishing_rate Numeric. Rate for Manifold Sculpting.
#'   Default in Tapkee: 0.99.
#' @param spe_global_strategy Logical. Use global strategy for SPE.
#'   Default in Tapkee: \code{TRUE}.
#' @param spe_num_updates Integer. Number of updates for SPE.
#'   Default in Tapkee: 100.
#' @param spe_tolerance Numeric. Convergence tolerance for SPE.
#'   Default in Tapkee: 1e-9.
#' @param nullspace_shift Numeric. Diagonal shift for eigenproblems.
#'   Default in Tapkee: 1e-9.
#' @param klle_shift Numeric. KLLE regularizer.
#'   Default in Tapkee: 1e-3.
#' @param fa_epsilon Numeric. Factor Analysis convergence epsilon.
#'   Default in Tapkee: 1e-9.
#' @param check_connectivity Logical. Check neighborhood graph connectivity.
#'   Default in Tapkee: \code{TRUE}.
#'
#' @return A numeric matrix with \code{ncol(data)} rows and
#'   \code{target_dimension} columns. Each row is the embedded
#'   representation of the corresponding input sample.
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(300), nrow = 3, ncol = 100)
#' embedding <- tapkee_embed(X, method = "pca", target_dimension = 2L)
#' plot(embedding[, 1], embedding[, 2], main = "PCA embedding")
#'
#' @export
tapkee_embed <- function(
    data,
    method = "lle",
    num_neighbors = NULL,
    target_dimension = NULL,
    gaussian_kernel_width = NULL,
    landmark_ratio = NULL,
    max_iteration = NULL,
    diffusion_map_timesteps = NULL,
    sne_perplexity = NULL,
    sne_theta = NULL,
    squishing_rate = NULL,
    spe_global_strategy = NULL,
    spe_num_updates = NULL,
    spe_tolerance = NULL,
    nullspace_shift = NULL,
    klle_shift = NULL,
    fa_epsilon = NULL,
    check_connectivity = NULL
) {
  if (!is.matrix(data) || !is.numeric(data)) {
    stop("'data' must be a numeric matrix")
  }
  if (!is.character(method) || length(method) != 1L) {
    stop("'method' must be a single character string")
  }

  if (!is.null(num_neighbors)) num_neighbors <- as.integer(num_neighbors)
  if (!is.null(target_dimension)) target_dimension <- as.integer(target_dimension)
  if (!is.null(max_iteration)) max_iteration <- as.integer(max_iteration)
  if (!is.null(diffusion_map_timesteps)) {
    diffusion_map_timesteps <- as.integer(diffusion_map_timesteps)
  }
  if (!is.null(spe_num_updates)) spe_num_updates <- as.integer(spe_num_updates)

  tapkee_embed_cpp(
    data = data,
    method = method,
    num_neighbors = num_neighbors,
    target_dimension = target_dimension,
    gaussian_kernel_width = gaussian_kernel_width,
    landmark_ratio = landmark_ratio,
    max_iteration = max_iteration,
    diffusion_map_timesteps = diffusion_map_timesteps,
    sne_perplexity = sne_perplexity,
    sne_theta = sne_theta,
    squishing_rate = squishing_rate,
    spe_global_strategy = spe_global_strategy,
    spe_num_updates = spe_num_updates,
    spe_tolerance = spe_tolerance,
    nullspace_shift = nullspace_shift,
    klle_shift = klle_shift,
    fa_epsilon = fa_epsilon,
    check_connectivity = check_connectivity
  )
}
