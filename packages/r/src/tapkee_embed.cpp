// RcppEigen must be included first to set up Eigen
#include <RcppEigen.h>

// Tell tapkee to use our bridge header instead of its own Eigen include.
// This avoids EIGEN_RUNTIME_NO_MALLOC and lets RcppEigen's Eigen take effect.
#define TAPKEE_EIGEN_INCLUDE_FILE "tapkee_eigen_rcpp.h"

// Use R's RNG instead of std::rand() (CRAN policy: no system RNGs)
#define CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION ((int)(R_unif_index(2147483647)))
#define CUSTOM_UNIFORM_RANDOM_FUNCTION R::runif(0.0, 1.0)
#define CUSTOM_GAUSSIAN_RANDOM_FUNCTION R::rnorm(0.0, 1.0)

#include <stichwort/parameter.hpp>
#include <tapkee/chain_interface.hpp>
#include <tapkee/defines.hpp>
#include <tapkee/defines/methods.hpp>
#include <tapkee/defines/keywords.hpp>

#include <map>
#include <string>

using stichwort::Parameter;
using tapkee::DimensionReductionMethod;
using tapkee::ParametersSet;
using tapkee::DenseMatrix;

// Custom logger that routes tapkee messages through R's messaging system
class RLoggerImplementation : public tapkee::LoggerImplementation {
  public:
    RLoggerImplementation() {}
    virtual ~RLoggerImplementation() {}
    virtual void message_info(const std::string& msg) { REprintf("[info] %s\n", msg.c_str()); }
    virtual void message_warning(const std::string& msg) { REprintf("[warning] %s\n", msg.c_str()); }
    virtual void message_debug(const std::string& msg) { REprintf("[debug] %s\n", msg.c_str()); }
    virtual void message_error(const std::string& msg) { REprintf("[error] %s\n", msg.c_str()); }
    virtual void message_benchmark(const std::string& msg) { REprintf("[benchmark] %s\n", msg.c_str()); }
};

static bool logger_initialized = false;

static void ensure_r_logger() {
    if (!logger_initialized) {
        tapkee::Logging::instance().set_logger_impl(new RLoggerImplementation());
        logger_initialized = true;
    }
}

// Self-contained method name parser (avoids pulling in cli/util.hpp)
static const std::map<std::string, DimensionReductionMethod> METHODS = {
    {"ltsa", tapkee::KernelLocalTangentSpaceAlignment},
    {"local_tangent_space_alignment", tapkee::KernelLocalTangentSpaceAlignment},
    {"lle", tapkee::KernelLocallyLinearEmbedding},
    {"locally_linear_embedding", tapkee::KernelLocallyLinearEmbedding},
    {"hlle", tapkee::HessianLocallyLinearEmbedding},
    {"hessian_locally_linear_embedding", tapkee::HessianLocallyLinearEmbedding},
    {"mds", tapkee::MultidimensionalScaling},
    {"multidimensional_scaling", tapkee::MultidimensionalScaling},
    {"l-mds", tapkee::LandmarkMultidimensionalScaling},
    {"landmark_multidimensional_scaling", tapkee::LandmarkMultidimensionalScaling},
    {"isomap", tapkee::Isomap},
    {"l-isomap", tapkee::LandmarkIsomap},
    {"landmark_isomap", tapkee::LandmarkIsomap},
    {"dm", tapkee::DiffusionMap},
    {"diffusion_map", tapkee::DiffusionMap},
    {"kpca", tapkee::KernelPrincipalComponentAnalysis},
    {"kernel_pca", tapkee::KernelPrincipalComponentAnalysis},
    {"pca", tapkee::PrincipalComponentAnalysis},
    {"random_projection", tapkee::RandomProjection},
    {"ra", tapkee::RandomProjection},
    {"la", tapkee::LaplacianEigenmaps},
    {"laplacian_eigenmaps", tapkee::LaplacianEigenmaps},
    {"lpp", tapkee::LocalityPreservingProjections},
    {"locality_preserving_projections", tapkee::LocalityPreservingProjections},
    {"npe", tapkee::NeighborhoodPreservingEmbedding},
    {"neighborhood_preserving_embedding", tapkee::NeighborhoodPreservingEmbedding},
    {"lltsa", tapkee::LinearLocalTangentSpaceAlignment},
    {"linear_local_tangent_space_alignment", tapkee::LinearLocalTangentSpaceAlignment},
    {"spe", tapkee::StochasticProximityEmbedding},
    {"stochastic_proximity_embedding", tapkee::StochasticProximityEmbedding},
    {"fa", tapkee::FactorAnalysis},
    {"factor_analysis", tapkee::FactorAnalysis},
    {"t-sne", tapkee::tDistributedStochasticNeighborEmbedding},
    {"t-stochastic_proximity_embedding", tapkee::tDistributedStochasticNeighborEmbedding},
    {"manifold_sculpting", tapkee::ManifoldSculpting},
    {"passthru", tapkee::PassThru},
};

static DimensionReductionMethod parse_method(const std::string& name) {
    auto it = METHODS.find(name);
    if (it != METHODS.end()) {
        return it->second;
    }
    Rcpp::stop("Unknown method: '%s'. Use one of: lle, isomap, pca, t-sne, mds, "
               "dm, kpca, la, lpp, npe, ltsa, lltsa, hlle, l-mds, l-isomap, "
               "spe, fa, random_projection, manifold_sculpting, passthru",
               name.c_str());
    return tapkee::PassThru; // unreachable
}

// [[Rcpp::export]]
Eigen::MatrixXd tapkee_embed_cpp(
    const Eigen::MatrixXd& data,
    const std::string& method,
    Rcpp::Nullable<int> num_neighbors,
    Rcpp::Nullable<int> target_dimension,
    Rcpp::Nullable<double> gaussian_kernel_width,
    Rcpp::Nullable<double> landmark_ratio,
    Rcpp::Nullable<int> max_iteration,
    Rcpp::Nullable<int> diffusion_map_timesteps,
    Rcpp::Nullable<double> sne_perplexity,
    Rcpp::Nullable<double> sne_theta,
    Rcpp::Nullable<double> squishing_rate,
    Rcpp::Nullable<bool> spe_global_strategy,
    Rcpp::Nullable<int> spe_num_updates,
    Rcpp::Nullable<double> spe_tolerance,
    Rcpp::Nullable<double> nullspace_shift,
    Rcpp::Nullable<double> klle_shift,
    Rcpp::Nullable<double> fa_epsilon,
    Rcpp::Nullable<bool> check_connectivity
) {
    ensure_r_logger();

    ParametersSet params;
    params.add(Parameter::create("dimension reduction method", parse_method(method)));

    if (num_neighbors.isNotNull())
        params.add(Parameter::create("number of neighbors", Rcpp::as<int>(num_neighbors)));
    if (target_dimension.isNotNull())
        params.add(Parameter::create("target dimension", Rcpp::as<int>(target_dimension)));
    if (gaussian_kernel_width.isNotNull())
        params.add(Parameter::create("the width of the gaussian kernel",
                   Rcpp::as<double>(gaussian_kernel_width)));
    if (landmark_ratio.isNotNull())
        params.add(Parameter::create("ratio of landmark points",
                   Rcpp::as<double>(landmark_ratio)));
    if (max_iteration.isNotNull())
        params.add(Parameter::create("maximal iteration", Rcpp::as<int>(max_iteration)));
    if (diffusion_map_timesteps.isNotNull())
        params.add(Parameter::create("diffusion map timesteps",
                   Rcpp::as<int>(diffusion_map_timesteps)));
    if (sne_perplexity.isNotNull())
        params.add(Parameter::create("SNE perplexity", Rcpp::as<double>(sne_perplexity)));
    if (sne_theta.isNotNull())
        params.add(Parameter::create("SNE theta", Rcpp::as<double>(sne_theta)));
    if (squishing_rate.isNotNull())
        params.add(Parameter::create("squishing rate", Rcpp::as<double>(squishing_rate)));
    if (spe_global_strategy.isNotNull())
        params.add(Parameter::create("SPE global strategy",
                   Rcpp::as<bool>(spe_global_strategy)));
    if (spe_num_updates.isNotNull())
        params.add(Parameter::create("SPE number of updates",
                   Rcpp::as<int>(spe_num_updates)));
    if (spe_tolerance.isNotNull())
        params.add(Parameter::create("SPE tolerance", Rcpp::as<double>(spe_tolerance)));
    if (nullspace_shift.isNotNull())
        params.add(Parameter::create("diagonal shift of nullspace",
                   Rcpp::as<double>(nullspace_shift)));
    if (klle_shift.isNotNull())
        params.add(Parameter::create("KLLE regularizer", Rcpp::as<double>(klle_shift)));
    if (fa_epsilon.isNotNull())
        params.add(Parameter::create("epsilon of FA", Rcpp::as<double>(fa_epsilon)));
    if (check_connectivity.isNotNull())
        params.add(Parameter::create("check connectivity",
                   Rcpp::as<bool>(check_connectivity)));

    DenseMatrix tapkee_data = data;
    return tapkee::with(params).embedUsing(tapkee_data).embedding;
}
