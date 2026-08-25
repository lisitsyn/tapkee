#include <cli/util.hpp>

#include <stichwort/parameter.hpp>

#include <tapkee/chain_interface.hpp>
#include <tapkee/defines.hpp>
#include <tapkee/defines/keywords.hpp>
#include <tapkee/defines/methods.hpp>

#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <emscripten/val.h>

#include <string>
#include <vector>

using emscripten::val;

using stichwort::Parameter;

using tapkee::DenseMatrix;
using tapkee::ParametersSet;
using tapkee::TapkeeOutput;
using tapkee::with;

namespace
{

[[noreturn]] void throw_js_error(const std::string& message)
{
    EM_ASM({ throw new Error(UTF8ToString($0)); }, message.c_str());
    __builtin_unreachable();
}

template <typename T> void add_if_present(ParametersSet& params, const val& options, const char* key, const char* name)
{
    const val v = options[key];
    if (!v.isUndefined() && !v.isNull())
        params.add(Parameter::create(name, v.as<T>()));
}

ParametersSet parse_options(const val& options)
{
    ParametersSet params;

    const val method = options["method"];
    const std::string method_name = (method.isUndefined() || method.isNull()) ? "lle" : method.as<std::string>();
    params.add(Parameter::create("dimension reduction method", parse_reduction_method(method_name)));

    const val neighbors_method = options["neighborsMethod"];
    if (!neighbors_method.isUndefined() && !neighbors_method.isNull())
        params.add(Parameter::create("nearest neighbors method",
                                     parse_multiple(NEIGHBORS_METHODS, neighbors_method.as<std::string>())));

    const val eigen_method = options["eigenMethod"];
    if (!eigen_method.isUndefined() && !eigen_method.isNull())
        params.add(
            Parameter::create("eigendecomposition method", parse_multiple(EIGEN_METHODS, eigen_method.as<std::string>())));

    add_if_present<int>(params, options, "numNeighbors", "number of neighbors");
    add_if_present<int>(params, options, "targetDimension", "target dimension");
    add_if_present<double>(params, options, "gaussianKernelWidth", "the width of the gaussian kernel");
    add_if_present<double>(params, options, "landmarkRatio", "ratio of landmark points");
    add_if_present<int>(params, options, "maxIteration", "maximal iteration");
    add_if_present<int>(params, options, "diffusionMapTimesteps", "diffusion map timesteps");
    add_if_present<double>(params, options, "snePerplexity", "SNE perplexity");
    add_if_present<double>(params, options, "sneTheta", "SNE theta");
    add_if_present<double>(params, options, "squishingRate", "squishing rate");
    add_if_present<bool>(params, options, "speGlobalStrategy", "SPE global strategy");
    add_if_present<int>(params, options, "speNumUpdates", "SPE number of updates");
    add_if_present<double>(params, options, "speTolerance", "SPE tolerance");
    add_if_present<double>(params, options, "nullspaceShift", "diagonal shift of nullspace");
    add_if_present<double>(params, options, "klleShift", "KLLE regularizer");
    add_if_present<double>(params, options, "faEpsilon", "epsilon of FA");
    add_if_present<bool>(params, options, "checkConnectivity", "check connectivity");

    return params;
}

} // namespace

// data is a flat array, point-major: [x0,y0,z0, x1,y1,z1, ...];
// returns {embedding: number[], rows: nPoints, cols: targetDimension}
val embed(const val& js_data, int n_points, int n_dims, const val& options)
{
    try
    {
        const std::vector<double> data = emscripten::convertJSArrayToNumberVector<double>(js_data);
        if (static_cast<long long>(data.size()) != static_cast<long long>(n_points) * n_dims)
            throw std::runtime_error("Data length does not match nPoints * nDims");

        DenseMatrix features(n_dims, n_points);
        for (int i = 0; i < n_points; ++i)
            for (int j = 0; j < n_dims; ++j)
                features(j, i) = data[static_cast<std::size_t>(i) * n_dims + j];

        const TapkeeOutput output = with(parse_options(options)).embedUsing(features);

        const int target_dimension = static_cast<int>(output.embedding.cols());
        std::vector<double> flat(static_cast<std::size_t>(n_points) * target_dimension);
        for (int i = 0; i < n_points; ++i)
            for (int j = 0; j < target_dimension; ++j)
                flat[static_cast<std::size_t>(i) * target_dimension + j] = output.embedding(i, j);

        val result = val::object();
        result.set("embedding", val::array(flat.begin(), flat.end()));
        result.set("rows", n_points);
        result.set("cols", target_dimension);
        return result;
    }
    catch (const std::exception& e)
    {
        throw_js_error(e.what());
    }
}

EMSCRIPTEN_BINDINGS(tapkee_module)
{
    emscripten::function("embed", &embed);
}
