/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2024 Sergey Lisitsyn, Fernando Iglesias
 */

#include "util.hpp"
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <cxxopts.hpp>

#include <tapkee/callbacks/eigen_callbacks.hpp>
#include <tapkee/callbacks/precomputed_callbacks.hpp>
#include <tapkee/defines.hpp>
#include <tapkee/projection.hpp>
#include <tapkee/tapkee.hpp>
#include <tapkee/utils/logging.hpp>

#ifdef GIT_INFO
#define TAPKEE_CURRENT_GIT_INFO GIT_INFO
#else
#define TAPKEE_CURRENT_GIT_INFO "unknown"
#endif

using namespace Eigen;
using namespace std;

bool cancel()
{
    return false;
}

std::string shorter(const char* shorter)
{
    return std::string(shorter) + ",";
}

auto string_with_default(const char* defs) -> decltype(cxxopts::value<std::string>()->default_value(""))
{
    return cxxopts::value<std::string>()->default_value(defs);
}

auto double_with_default(const char* defs) -> decltype(cxxopts::value<double>()->default_value(""))
{
    return cxxopts::value<double>()->default_value(defs);
}

auto int_with_default(const char* defs) -> decltype(cxxopts::value<int>()->default_value(""))
{
    return cxxopts::value<int>()->default_value(defs);
}

std::vector<const char*> process_argv(int argc, const char** argv)
{
    std::vector<const char*> processed;
    for (int i=0; i<argc; ++i)
    {
        processed.push_back(argv[i]);
        #if defined(USE_SLASH_CLI_WINDOWS) && (defined(_WIN32) || defined(_WIN64))
        // rpplace -- and - with /
        #endif
    }
    return processed;
}

static const char* INPUT_FILE_KEYWORD = "input-file";
static const char* TRANSPOSE_INPUT_KEYWORD = "transpose-input";
static const char* TRANSPOSE_OUTPUT_KEYWORD = "transpose-output";
static const char* OUTPUT_FILE_KEYWORD = "output-file";
static const char* OUTPUT_PROJECTION_MATRIX_FILE_KEYWORD = "output-projection-matrix-file";
static const char* OUTPUT_PROJECTION_MEAN_FILE_KEYWORD = "output-projection-mean-file";
static const char* DELIMITER_KEYWORD = "delimiter";
static const char* HELP_KEYWORD = "help";
static const char* BENCHMARK_KEYWORD = "benchmark";
static const char* VERBOSE_KEYWORD = "verbose";
static const char* DEBUG_KEYWORD = "debug";
static const char* METHOD_KEYWORD = "method";
static const char* NEIGHBORS_METHOD_KEYWORD = "neighbors-method";
static const char* EIGEN_METHOD_KEYWORD = "eigen-method";
static const char* COMPUTATION_STRATEGY_KEYWORD = "computation-strategy";
static const char* TARGET_DIMENSION_KEYWORD = "target-dimension";
static const char* NUM_NEIGHBORS_KEYWORD = "num-neighbors";
static const char* GAUSSIAN_WIDTH_KEYWORD = "gaussian-width";
static const char* TIMESTEPS_KEYWORD = "timesteps";
static const char* SPE_LOCAL_KEYWORD = "spe-local";
static const char* EIGENSHIFT_KEYWORD = "eigenshift";;
static const char* LANDMARK_RATIO_KEYWORD = "landmark-ratio";
static const char* SPE_TOLERANCE_KEYWORD = "spe-tolerance";
static const char* SPE_NUM_UPDATES_KEYWORD = "spe-num-updates";
static const char* MAX_ITERS_KEYWORD = "max-iters";
static const char* FA_EPSILON_KEYWORD = "fa-epsilon";
static const char* SNE_PERPLEXITY_KEYWORD = "sne-perplexity";
static const char* SNE_THETA_KEYWORD = "sne-theta";
static const char* MS_SQUISHING_RATE_KEYWORD = "squishing-rate";
static const char* PRECOMPUTE_KEYWORD = "precompute";

int run(int argc, const char **argv)
{
    srand(static_cast<unsigned int>(time(NULL)));

    cxxopts::Options options("tapkee", "Tapkee: a tool for dimension reduction");
    auto processed_argv = process_argv(argc, argv);

    options
      .set_width(70)
      .set_tab_expansion()
      .add_options()
    (
     shorter("i") + INPUT_FILE_KEYWORD,
     "Input file",
     string_with_default("/dev/stdin")
    )
    (
     TRANSPOSE_INPUT_KEYWORD,
     "Transpose input file if set"
    )
    (
     TRANSPOSE_OUTPUT_KEYWORD,
     "Transpose output file if set"
    )
    (
     shorter("o") + OUTPUT_FILE_KEYWORD,
     "Output file",
     string_with_default("/dev/stdout")
    )
    (
     shorter("opmat") + OUTPUT_PROJECTION_MATRIX_FILE_KEYWORD,
     "Output file for the projection matrix",
     string_with_default("/dev/null")
    )
    (
     shorter("opmean") + OUTPUT_PROJECTION_MEAN_FILE_KEYWORD,
     "Output file for the mean of data",
     string_with_default("/dev/null")
    )
    (
     shorter("d") + DELIMITER_KEYWORD,
     "Delimiter",
     string_with_default(",")
    )
    (
     shorter("h") + HELP_KEYWORD,
     "Print usage"
    )
    (
     BENCHMARK_KEYWORD,
     "Output benchmark information"
    )
    (
     VERBOSE_KEYWORD,
     "Output more information"
    )
    (
     DEBUG_KEYWORD,
     "Output debug information"
    )
    (
     shorter("m") + METHOD_KEYWORD,
     "Dimension reduction method (default locally_linear_embedding). \n One of the following: \n"
     "locally_linear_embedding (lle), neighborhood_preserving_embedding (npe), \n"
     "local_tangent_space_alignment (ltsa), linear_local_tangent_space_alignment (lltsa), \n"
     "hessian_locally_linear_embedding (hlle), laplacian_eigenmaps (la), locality_preserving_projections (lpp), \n"
     "diffusion_map (dm), isomap, landmark_isomap (l-isomap), multidimensional_scaling (mds), \n"
     "landmark_multidimensional_scaling (l-mds), stochastic_proximity_embedding (spe), \n"
     "kernel_pca (kpca), pca, random_projection (ra), factor_analysis (fa), \n"
     "t-stochastic_neighborhood_embedding (t-sne), manifold_sculpting (ms).",
     string_with_default("locally_linear_embedding")
    )
    (
     shorter("nm") + NEIGHBORS_METHOD_KEYWORD,
     "Neighbors search method (default is 'covertree' if available, 'vptree' otherwise). One of the following: "
     "brute,vptree"
#ifdef TAPKEE_USE_LGPL_COVERTREE
     ",covertree"
#endif
     ".",
#ifdef TAPKEE_USE_LGPL_COVERTREE
     string_with_default("covertree")
#else
     string_with_default("vptree")
#endif
    )
    (
     shorter("em") + EIGEN_METHOD_KEYWORD,
     "Eigendecomposition method (default is 'arpack' if available, 'dense' otherwise). One of the following: "
#ifdef TAPKEE_WITH_ARPACK
     "arpack, "
#endif
     "randomized, dense.",
#ifdef TAPKEE_WITH_ARPACK
     string_with_default("arpack")
#else
     string_with_default("dense")
#endif
    )
    (
     shorter("cs") + COMPUTATION_STRATEGY_KEYWORD,
     "Computation strategy (default is 'cpu'). One of the following: "
#ifdef TAPKEE_WITH_VIENNACL
     "opencl, "
#endif
     "cpu.",
     string_with_default("cpu")
    )
    (
     shorter("td") + TARGET_DIMENSION_KEYWORD,
     "Target dimension",
     int_with_default("2")
    )
    (
     shorter("k") + NUM_NEIGHBORS_KEYWORD,
     "Number of neighbors",
     int_with_default("10")
    )
    (
     shorter("gw") + GAUSSIAN_WIDTH_KEYWORD,
     "Width of gaussian kernel",
     double_with_default("1.0")
    )
    (
     TIMESTEPS_KEYWORD,
     "Number of timesteps for diffusion map",
     int_with_default("1")
    )
    (
     EIGENSHIFT_KEYWORD,
     "Regularization diagonal shift for weight matrix",
     double_with_default("1e-9")
    )
    (
     LANDMARK_RATIO_KEYWORD,
     "Ratio of landmarks. Should be in (0,1) range (0.2 means 20%)",
     double_with_default("0.2")
    )
    (
     SPE_LOCAL_KEYWORD,
     "Local strategy in SPE (default is global)"
    )
    (
     SPE_TOLERANCE_KEYWORD,
     "Tolerance for SPE",
     double_with_default("1e-5")
    )
    (
     SPE_NUM_UPDATES_KEYWORD,
     "Number of SPE updates",
     int_with_default("100")
    )
    (
     MAX_ITERS_KEYWORD,
     "Maximum number of iterations",
     int_with_default("1000")
    )
    (
     FA_EPSILON_KEYWORD,
     "FA convergence threshold",
     double_with_default("1e-5")
    )
    (
     SNE_PERPLEXITY_KEYWORD,
     "Perplexity for the t-SNE algorithm",
     double_with_default("30.0")
    )
    (
     SNE_THETA_KEYWORD,
     "Theta for the t-SNE algorithm",
     double_with_default("0.5")
    )
    (
     MS_SQUISHING_RATE_KEYWORD,
     "Squishing rate of the Manifold Sculpting algorithm",
     double_with_default("0.99")
    )
    (
     PRECOMPUTE_KEYWORD,
     "Whether distance and kernel matrices should be precomputed (default false)"
    )
    ;

    auto opt = options.parse(processed_argv.size(), &processed_argv[0]);

    if (opt.count(HELP_KEYWORD))
    {
        std::cout << options.help() << std::endl << std::endl;
        std::cout << "Git version " TAPKEE_CURRENT_GIT_INFO << std::endl << std::endl;
        std::cout << "Example: " << std::endl <<
            "Run locally linear embedding with k=10 with arpack " << std::endl <<
            "eigensolver on data from input.dat saving embedding to output.dat" << std::endl << std::endl <<
            "tapkee -i input.dat -o output.dat --method lle --eigen-method arpack -k 10" << std::endl << std::endl;

        std::cout <<
            "Copyright (C) 2012-2024 Sergey Lisitsyn <lisitsyn@hey.com>, Fernando Iglesias <fernando.iglesiasg@gmail.com>" << std::endl <<
            "This is free software: you are free to change and redistribute it" << std::endl <<
            "There is NO WARRANTY, to the extent permitted by law." << std::endl;
        return 1;
    }
    if (opt.count(VERBOSE_KEYWORD))
    {
        tapkee::LoggingSingleton::instance().enable_info();
    }
    if (opt.count(DEBUG_KEYWORD))
    {
        tapkee::LoggingSingleton::instance().enable_debug();
        tapkee::LoggingSingleton::instance().message_info("Debug messages enabled");
    }

    if (opt.count(BENCHMARK_KEYWORD))
    {
        tapkee::LoggingSingleton::instance().enable_benchmark();
        tapkee::LoggingSingleton::instance().message_info("Benchmarking enabled");
    }

    tapkee::DimensionReductionMethod tapkee_method;
    {
        string method = opt[METHOD_KEYWORD].as<std::string>();
        try
        {
            tapkee_method = parse_reduction_method(method.c_str());
        }
        catch (const std::exception &)
        {
            tapkee::LoggingSingleton::instance().message_error(string("Unknown method ") + method);
            return 1;
        }
    }

    tapkee::NeighborsMethod tapkee_neighbors_method = tapkee::Brute;
    {
        string method = opt[NEIGHBORS_METHOD_KEYWORD].as<std::string>();
        try
        {
            tapkee_neighbors_method = parse_neighbors_method(method.c_str());
        }
        catch (const std::exception &)
        {
            tapkee::LoggingSingleton::instance().message_error(string("Unknown neighbors method ") + method);
            return 1;
        }
    }
    tapkee::EigenMethod tapkee_eigen_method = tapkee::Dense;
    {
        string method = opt[EIGEN_METHOD_KEYWORD].as<std::string>();
        try
        {
            tapkee_eigen_method = parse_eigen_method(method.c_str());
        }
        catch (const std::exception &)
        {
            tapkee::LoggingSingleton::instance().message_error(string("Unknown eigendecomposition method ") + method);
            return 1;
        }
    }
    tapkee::ComputationStrategy tapkee_computation_strategy = tapkee::HomogeneousCPUStrategy;
    {
        string method = opt[COMPUTATION_STRATEGY_KEYWORD].as<std::string>();
        try
        {
            tapkee_computation_strategy = parse_computation_strategy(method.c_str());
        }
        catch (const std::exception &)
        {
            tapkee::LoggingSingleton::instance().message_error(string("Unknown computation strategy ") + method);
            return 1;
        }
    }

    int target_dim = opt[TARGET_DIMENSION_KEYWORD].as<int>();
    if (target_dim < 0)
    {
        tapkee::LoggingSingleton::instance().message_error(
            "Negative target dimensionality is not possible in current circumstances. "
            "Please visit other universe");
        return 1;
    }

    int k = opt[NUM_NEIGHBORS_KEYWORD].as<int>();
    if (k < 3)
    {
        tapkee::LoggingSingleton::instance().message_error(
            "The provided number of neighbors is too small, consider at least 3.");
        return 1;
    }
    double width = opt[GAUSSIAN_WIDTH_KEYWORD].as<double>();
    if (width < 0.0)
    {
        tapkee::LoggingSingleton::instance().message_error("Width of the gaussian kernel is negative.");
        return 1;
    }
    int timesteps = opt[TIMESTEPS_KEYWORD].as<int>();
    if (timesteps < 0)
    {
        tapkee::LoggingSingleton::instance().message_error("Number of timesteps is negative.");
        return 1;
    }
    double eigenshift = opt[EIGENSHIFT_KEYWORD].as<double>();
    double landmark_rt = opt[LANDMARK_RATIO_KEYWORD].as<double>();
    bool spe_global = opt.count(SPE_LOCAL_KEYWORD);
    double spe_tol = opt[SPE_TOLERANCE_KEYWORD].as<double>();
    int spe_num_upd = opt[SPE_NUM_UPDATES_KEYWORD].as<int>();
    int max_iters = opt[MAX_ITERS_KEYWORD].as<int>();
    double fa_eps = opt[FA_EPSILON_KEYWORD].as<double>();
    double perplexity = opt[SNE_PERPLEXITY_KEYWORD].as<double>();
    double theta = opt[SNE_THETA_KEYWORD].as<double>();
    double squishing = opt[MS_SQUISHING_RATE_KEYWORD].as<double>();

    // Load data
    string input_filename = opt[INPUT_FILE_KEYWORD].as<std::string>();
    string output_filename = opt[OUTPUT_FILE_KEYWORD].as<std::string>();

    bool output_projection = false;
    std::string output_matrix_filename = opt[OUTPUT_PROJECTION_MATRIX_FILE_KEYWORD].as<std::string>();
    std::string output_mean_filename = opt[OUTPUT_PROJECTION_MEAN_FILE_KEYWORD].as<std::string>();
    if (opt.count(OUTPUT_PROJECTION_MATRIX_FILE_KEYWORD) &&
        opt.count(OUTPUT_PROJECTION_MEAN_FILE_KEYWORD))
    {
        output_projection = true;
    }

    ifstream ifs(input_filename.c_str());
    ofstream ofs(output_filename.c_str());
    ofstream ofs_matrix(output_matrix_filename.c_str());
    ofstream ofs_mean(output_mean_filename.c_str());

    std::string delimiter = opt[DELIMITER_KEYWORD].as<std::string>();

    tapkee::DenseMatrix input_data = read_data(ifs, delimiter[0]);
    if (!opt.count(TRANSPOSE_INPUT_KEYWORD))
    {
        input_data.transposeInPlace();
    }

    std::stringstream ss;
    ss << "Data contains " << input_data.cols() << " feature vectors with dimension of " << input_data.rows();
    tapkee::LoggingSingleton::instance().message_info(ss.str());

    tapkee::TapkeeOutput output;

    tapkee::ParametersSet parameters =
        tapkee::kwargs[(tapkee::method = tapkee_method, tapkee::computation_strategy = tapkee_computation_strategy,
                        tapkee::eigen_method = tapkee_eigen_method, tapkee::neighbors_method = tapkee_neighbors_method,
                        tapkee::num_neighbors = k, tapkee::target_dimension = target_dim,
                        tapkee::diffusion_map_timesteps = timesteps, tapkee::gaussian_kernel_width = width,
                        tapkee::max_iteration = max_iters, tapkee::spe_global_strategy = spe_global,
                        tapkee::spe_num_updates = spe_num_upd, tapkee::spe_tolerance = spe_tol,
                        tapkee::landmark_ratio = landmark_rt, tapkee::nullspace_shift = eigenshift,
                        tapkee::check_connectivity = true, tapkee::fa_epsilon = fa_eps,
                        tapkee::sne_perplexity = perplexity, tapkee::sne_theta = theta,
                        tapkee::squishing_rate = squishing)];

    if (opt.count(PRECOMPUTE_KEYWORD))
    {
        vector<tapkee::IndexType> indices(input_data.cols());
        for (tapkee::IndexType i = 0; i < input_data.cols(); ++i)
            indices[i] = i;

        tapkee::DenseMatrix distance_matrix;
        tapkee::DenseMatrix kernel_matrix;
        {
            if (method_needs_distance(tapkee_method))
            {
                tapkee::tapkee_internal::timed_context context("[+] Distance matrix computation");
                distance_matrix = matrix_from_callback(static_cast<tapkee::IndexType>(input_data.cols()),
                                                    tapkee::eigen_distance_callback(input_data));
            }
            if (method_needs_kernel(tapkee_method))
            {
                tapkee::tapkee_internal::timed_context context("[+] Kernel matrix computation");
                kernel_matrix = matrix_from_callback(static_cast<tapkee::IndexType>(input_data.cols()),
                                                    tapkee::eigen_kernel_callback(input_data));
            }
        }
        tapkee::precomputed_distance_callback dcb(distance_matrix);
        tapkee::precomputed_kernel_callback kcb(kernel_matrix);
        tapkee::eigen_features_callback fcb(input_data);

        output = tapkee::initialize()
                    .withParameters(parameters)
                    .withKernel(kcb)
                    .withDistance(dcb)
                    .withFeatures(fcb)
                    .embedRange(indices.begin(), indices.end());
    }
    else
    {
        output = tapkee::initialize().withParameters(parameters).embedUsing(input_data);
    }
    // Save obtained data
    if (opt.count(TRANSPOSE_OUTPUT_KEYWORD))
    {
        output.embedding.transposeInPlace();
    }
    write_matrix(&output.embedding, ofs, delimiter[0]);
    ofs.close();

    if (output_projection && output.projection.implementation)
    {
        tapkee::MatrixProjectionImplementation* matrix_projection =
            dynamic_cast<tapkee::MatrixProjectionImplementation *>(output.projection.implementation.get());
        if (!matrix_projection)
        {
            tapkee::LoggingSingleton::instance().message_error("Projection function unavailable");
            return 1;
        }
    }
    ofs_matrix.close();
    ofs_mean.close();
    return 0;
}

int main(int argc, const char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception &exc)
    {
        std::cerr << "Some error occured: " << exc.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occured" << std::endl;
        return 1;
    }
}
