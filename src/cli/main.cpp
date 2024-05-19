/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2024 Sergey Lisitsyn, Fernando Iglesias
 */

#include "util.hpp"
#include <algorithm>
#include <iterator>
#include <string>
#include <type_traits>
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

std::string either(const char* shorter_keyword, const char* keyword)
{
    return std::string(shorter_keyword) + "," + keyword;
}

template<typename T> auto with_default(T defs)
{
    if constexpr (std::is_same_v<std::string, T>)
        return cxxopts::value<T>()->default_value(defs);
    else
        return cxxopts::value<T>()->default_value(std::to_string(defs));
}

static const char* INPUT_FILE_KEYWORD = "input-file";
static const char* INPUT_FILE_DESCRIPTION = "Input filename to be used. Can be any file that can be opened for reading by the program. Expects delimiter-separated matrix of real values. See transposing options for more details on rows and columns.";

static const char* TRANSPOSE_INPUT_KEYWORD = "transpose-input";
static const char* TRANSPOSE_INPUT_DESCRIPTION = "Whether input file should be considered transposed. By default a line means a row in a matrix (a single vector to be embedded).";

static const char* TRANSPOSE_OUTPUT_KEYWORD = "transpose-output";
static const char* TRANSPOSE_OUTPUT_DESCRIPTION = "Whether output file should be transposed. By default a line would be a row of embedding matrix (a single embedding vector)";

static const char* OUTPUT_FILE_KEYWORD = "output-file";
static const char* OUTPUT_FILE_DESCRIPTION = "Output filename to be used. Can be any file that can be opened for writing by the program";

static const char* OUTPUT_PROJECTION_MATRIX_FILE_KEYWORD = "output-projection-matrix-file";
static const char* OUTPUT_PROJECTION_MATRIX_FILE_DESCRIPTION = "Filename to store the projection matrix calculated by the selected algorithm. Usually supported by linear algorithms such as PCA.";

static const char* OUTPUT_PROJECTION_MEAN_FILE_KEYWORD = "output-projection-mean-file";
static const char* OUTPUT_PROJECTION_MEAN_FILE_DESCRIPTION = "Filename to store the mean vector calculated by the selected algorithm. Usually supported by linear algorithms such as PCA";

static const char* DELIMITER_KEYWORD = "delimiter";
static const char* DELIMITER_DESCRIPTION = "Delimiter to be used in reading and writing matrices";

static const char* HELP_KEYWORD = "help";
static const char* HELP_DESCRIPTION = "Print usage of the program";

static const char* BENCHMARK_KEYWORD = "benchmark";
static const char* BENCHMARK_DESCRIPTION = "Output benchmarking information about the time of algorithm steps";

static const char* VERBOSE_KEYWORD = "verbose";
static const char* VERBOSE_DESCRIPTION = "Be more verbose in logging";

static const char* DEBUG_KEYWORD = "debug";
static const char* DEBUG_DESCRIPTION = "Output debugging information such as intermediary steps, parameters, and other internals";

static const char* METHOD_KEYWORD = "method";
static const std::string METHOD_DESCRIPTION = "Dimension reduction method. One of the following: " +
    comma_separated_keys(DIMENSION_REDUCTION_METHODS.begin(), DIMENSION_REDUCTION_METHODS.end());

static const char* NEIGHBORS_METHOD_KEYWORD = "neighbors-method";
static const std::string NEIGHBORS_METHOD_DESCRIPTION = "Neighbors search method. One of the following: " +
    comma_separated_keys(NEIGHBORS_METHODS.begin(), NEIGHBORS_METHODS.end());

static const char* EIGEN_METHOD_KEYWORD = "eigen-method";
static const std::string EIGEN_METHOD_DESCRIPTION = "Eigendecomposition method. One of the following: " +
    comma_separated_keys(EIGEN_METHODS.begin(), EIGEN_METHODS.end());

static const char* COMPUTATION_STRATEGY_KEYWORD = "computation-strategy";
static const std::string COMPUTATION_STRATEGY_DESCRIPTION = "Computation strategy. One of the following: " +
    comma_separated_keys(COMPUTATION_STRATEGIES.begin(), COMPUTATION_STRATEGIES.end());

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

    cxxopts::Options options("tapkee", "Tapkee: a tool for dimensionality reduction.");

    using namespace std::string_literals;

    options
      .set_width(70)
      .set_tab_expansion()
      .add_options()
    (
     either("i", INPUT_FILE_KEYWORD),
     INPUT_FILE_DESCRIPTION,
     with_default("/dev/stdin"s)
    )
    (
     TRANSPOSE_INPUT_KEYWORD,
     TRANSPOSE_INPUT_DESCRIPTION
    )
    (
     TRANSPOSE_OUTPUT_KEYWORD,
     TRANSPOSE_OUTPUT_DESCRIPTION
    )
    (
     either("o", OUTPUT_FILE_KEYWORD),
     OUTPUT_FILE_DESCRIPTION,
     with_default("/dev/stdout"s)
    )
    (
     either("opmat", OUTPUT_PROJECTION_MATRIX_FILE_KEYWORD),
     OUTPUT_PROJECTION_MATRIX_FILE_DESCRIPTION,
     with_default("/dev/null"s)
    )
    (
     either("opmean", OUTPUT_PROJECTION_MEAN_FILE_KEYWORD),
     OUTPUT_PROJECTION_MEAN_FILE_DESCRIPTION,
     with_default("/dev/null"s)
    )
    (
     either("d", DELIMITER_KEYWORD),
     DELIMITER_DESCRIPTION,
     with_default(","s)
    )
    (
     either("h", HELP_KEYWORD),
     HELP_DESCRIPTION
    )
    (
     BENCHMARK_KEYWORD,
     BENCHMARK_DESCRIPTION
    )
    (
     VERBOSE_KEYWORD,
     VERBOSE_DESCRIPTION
    )
    (
     DEBUG_KEYWORD,
     DEBUG_DESCRIPTION
    )
    (
     either("m", METHOD_KEYWORD),
     METHOD_DESCRIPTION,
     with_default("locally_linear_embedding"s)
    )
    (
     either("nm", NEIGHBORS_METHOD_KEYWORD),
     NEIGHBORS_METHOD_DESCRIPTION,
#ifdef TAPKEE_USE_LGPL_COVERTREE
     with_default("covertree"s)
#else
     with_default("vptree"s)
#endif
    )
    (
     either("em", EIGEN_METHOD_KEYWORD),
     EIGEN_METHOD_DESCRIPTION,
#ifdef TAPKEE_WITH_ARPACK
     with_default("arpack"s)
#else
     with_default("dense"s)
#endif
    )
    (
     either("cs", COMPUTATION_STRATEGY_KEYWORD),
     COMPUTATION_STRATEGY_DESCRIPTION,
     with_default("cpu"s)
    )
    (
     either("td", TARGET_DIMENSION_KEYWORD),
     "Target dimension",
     with_default(2)
    )
    (
     either("k", NUM_NEIGHBORS_KEYWORD),
     "Number of neighbors",
     with_default(10)
    )
    (
     either("gw", GAUSSIAN_WIDTH_KEYWORD),
     "Width of gaussian kernel",
     with_default(1.0)
    )
    (
     TIMESTEPS_KEYWORD,
     "Number of timesteps for diffusion map",
     with_default(1)
    )
    (
     EIGENSHIFT_KEYWORD,
     "Regularization diagonal shift for weight matrix",
     with_default(1e-9)
    )
    (
     LANDMARK_RATIO_KEYWORD,
     "Ratio of landmarks. Should be in (0,1) range (0.2 means 20%)",
     with_default(0.2)
    )
    (
     SPE_LOCAL_KEYWORD,
     "Local strategy in SPE (default is global)"
    )
    (
     SPE_TOLERANCE_KEYWORD,
     "Tolerance for SPE",
     with_default(1e-5)
    )
    (
     SPE_NUM_UPDATES_KEYWORD,
     "Number of SPE updates",
     with_default(100)
    )
    (
     MAX_ITERS_KEYWORD,
     "Maximum number of iterations",
     with_default(1000)
    )
    (
     FA_EPSILON_KEYWORD,
     "FA convergence threshold",
     with_default(1e-5)
    )
    (
     SNE_PERPLEXITY_KEYWORD,
     "Perplexity for the t-SNE algorithm",
     with_default(30.0)
    )
    (
     SNE_THETA_KEYWORD,
     "Theta for the t-SNE algorithm",
     with_default(0.5)
    )
    (
     MS_SQUISHING_RATE_KEYWORD,
     "Squishing rate of the Manifold Sculpting algorithm",
     with_default(0.99)
    )
    (
     PRECOMPUTE_KEYWORD,
     "Whether distance and kernel matrices should be precomputed (default false)"
    )
    ;

    auto opt = options.parse(argc, argv);

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
        tapkee::Logging::instance().enable_info();
    }
    if (opt.count(DEBUG_KEYWORD))
    {
        tapkee::Logging::instance().enable_debug();
        tapkee::Logging::instance().message_info("Debug messages enabled");
    }

    if (opt.count(BENCHMARK_KEYWORD))
    {
        tapkee::Logging::instance().enable_benchmark();
        tapkee::Logging::instance().message_info("Benchmarking enabled");
    }

    tapkee::DimensionReductionMethod tapkee_method = tapkee::PassThru;
    {
        string method = opt[METHOD_KEYWORD].as<std::string>();
        try
        {
            tapkee_method = parse_multiple(DIMENSION_REDUCTION_METHODS, method);
        }
        catch (const std::exception & ex)
        {
            tapkee::Logging::instance().message_error(string("Unknown method ") + method);
            return 1;
        }
    }

    tapkee::NeighborsMethod tapkee_neighbors_method = tapkee::Brute;
    {
        string method = opt[NEIGHBORS_METHOD_KEYWORD].as<std::string>();
        try
        {
            tapkee_neighbors_method = parse_multiple(NEIGHBORS_METHODS, method);
        }
        catch (const std::exception &)
        {
            tapkee::Logging::instance().message_error(string("Unknown neighbors method ") + method);
            return 1;
        }
    }
    tapkee::EigenMethod tapkee_eigen_method = tapkee::Dense;
    {
        string method = opt[EIGEN_METHOD_KEYWORD].as<std::string>();
        try
        {
            tapkee_eigen_method = parse_multiple(EIGEN_METHODS, method);
        }
        catch (const std::exception &)
        {
            tapkee::Logging::instance().message_error(string("Unknown eigendecomposition method ") + method);
            return 1;
        }
    }
    tapkee::ComputationStrategy tapkee_computation_strategy = tapkee::HomogeneousCPUStrategy;
    {
        string method = opt[COMPUTATION_STRATEGY_KEYWORD].as<std::string>();
        try
        {
            tapkee_computation_strategy = parse_multiple(COMPUTATION_STRATEGIES, method);
        }
        catch (const std::exception &)
        {
            tapkee::Logging::instance().message_error(string("Unknown computation strategy ") + method);
            return 1;
        }
    }

    int target_dim = opt[TARGET_DIMENSION_KEYWORD].as<int>();
    if (target_dim < 0)
    {
        tapkee::Logging::instance().message_error(
            "Negative target dimensionality is not possible in current circumstances. "
            "Please visit other universe");
        return 1;
    }

    int k = opt[NUM_NEIGHBORS_KEYWORD].as<int>();
    if (k < 3)
    {
        tapkee::Logging::instance().message_error(
            "The provided number of neighbors is too small, consider at least 3.");
        return 1;
    }
    double width = opt[GAUSSIAN_WIDTH_KEYWORD].as<double>();
    if (width < 0.0)
    {
        tapkee::Logging::instance().message_error("Width of the gaussian kernel is negative.");
        return 1;
    }
    int timesteps = opt[TIMESTEPS_KEYWORD].as<int>();
    if (timesteps < 0)
    {
        tapkee::Logging::instance().message_error("Number of timesteps is negative.");
        return 1;
    }
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

    tapkee::Logging::instance().message_info(fmt::format("Data contains {} feature vectors with dimension of {}", input_data.cols(), input_data.rows()));

    tapkee::TapkeeOutput output;

    tapkee::ParametersSet parameters =
        tapkee::kwargs[(
                tapkee::method = tapkee_method,
                tapkee::computation_strategy = tapkee_computation_strategy,
                tapkee::eigen_method = tapkee_eigen_method,
                tapkee::neighbors_method = tapkee_neighbors_method,
                tapkee::num_neighbors = k,
                tapkee::target_dimension = target_dim,
                tapkee::diffusion_map_timesteps = timesteps,
                tapkee::gaussian_kernel_width = width,
                tapkee::max_iteration = opt[MAX_ITERS_KEYWORD].as<int>(),
                tapkee::spe_global_strategy = opt.count(SPE_LOCAL_KEYWORD),
                tapkee::spe_num_updates = opt[SPE_NUM_UPDATES_KEYWORD].as<int>(),
                tapkee::spe_tolerance = opt[SPE_TOLERANCE_KEYWORD].as<double>(),
                tapkee::landmark_ratio = opt[LANDMARK_RATIO_KEYWORD].as<double>(),
                tapkee::nullspace_shift = opt[EIGENSHIFT_KEYWORD].as<double>(),
                tapkee::check_connectivity = true,
                tapkee::fa_epsilon = opt[FA_EPSILON_KEYWORD].as<double>(),
                tapkee::sne_perplexity = opt[SNE_PERPLEXITY_KEYWORD].as<double>(),
                tapkee::sne_theta = opt[SNE_THETA_KEYWORD].as<double>(),
                tapkee::squishing_rate = opt[MS_SQUISHING_RATE_KEYWORD].as<double>()
        )];


    if (opt.count(PRECOMPUTE_KEYWORD))
    {
        vector<tapkee::IndexType> indices(input_data.cols());
        for (tapkee::IndexType i = 0; i < input_data.cols(); ++i)
            indices[i] = i;

        tapkee::DenseMatrix distance_matrix;
        tapkee::DenseMatrix kernel_matrix;
        {
            if (tapkee_method.needs_distance)
            {
                tapkee::tapkee_internal::timed_context context("[+] Distance matrix computation");
                distance_matrix = matrix_from_callback(static_cast<tapkee::IndexType>(input_data.cols()),
                                                       tapkee::eigen_distance_callback(input_data));
            }
            if (tapkee_method.needs_kernel)
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
            tapkee::Logging::instance().message_error("Projection function unavailable");
            return 1;
        }
        write_matrix(&matrix_projection->proj_mat, ofs_matrix, delimiter[0]);
        write_vector(&matrix_projection->mean_vec, ofs_mean);
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
