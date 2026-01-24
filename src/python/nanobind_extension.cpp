#include <cli/util.hpp>

#include <stichwort/parameter.hpp>

#include <tapkee/chain_interface.hpp>
#include <tapkee/defines.hpp>
#include <tapkee/defines/methods.hpp>
#include <tapkee/defines/keywords.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;

using stichwort::Parameter;

using tapkee::DimensionReductionMethod;
using tapkee::ParametersSet;
using tapkee::TapkeeOutput;
using tapkee::DenseMatrix;
using tapkee::with;

using tapkee::tapkee_internal::ParametersInitializedState;

// High-level embed function with keyword arguments
DenseMatrix embed(
    const DenseMatrix& data,
    const std::string& method,
    std::optional<int> num_neighbors,
    std::optional<int> target_dimension,
    std::optional<double> gaussian_kernel_width,
    std::optional<double> landmark_ratio,
    std::optional<int> max_iteration,
    std::optional<int> diffusion_map_timesteps,
    std::optional<double> sne_perplexity,
    std::optional<double> sne_theta,
    std::optional<double> squishing_rate,
    std::optional<bool> spe_global_strategy,
    std::optional<int> spe_num_updates,
    std::optional<double> spe_tolerance,
    std::optional<double> nullspace_shift,
    std::optional<double> klle_shift,
    std::optional<double> fa_epsilon,
    std::optional<bool> check_connectivity
) {
    ParametersSet params;

    // Required: method
    params.add(Parameter::create("dimension reduction method", parse_reduction_method(method)));

    // Optional parameters - only add if provided
    if (num_neighbors)
        params.add(Parameter::create("number of neighbors", *num_neighbors));
    if (target_dimension)
        params.add(Parameter::create("target dimension", *target_dimension));
    if (gaussian_kernel_width)
        params.add(Parameter::create("the width of the gaussian kernel", *gaussian_kernel_width));
    if (landmark_ratio)
        params.add(Parameter::create("ratio of landmark points", *landmark_ratio));
    if (max_iteration)
        params.add(Parameter::create("maximal iteration", *max_iteration));
    if (diffusion_map_timesteps)
        params.add(Parameter::create("diffusion map timesteps", *diffusion_map_timesteps));
    if (sne_perplexity)
        params.add(Parameter::create("SNE perplexity", *sne_perplexity));
    if (sne_theta)
        params.add(Parameter::create("SNE theta", *sne_theta));
    if (squishing_rate)
        params.add(Parameter::create("squishing rate", *squishing_rate));
    if (spe_global_strategy)
        params.add(Parameter::create("SPE global strategy", *spe_global_strategy));
    if (spe_num_updates)
        params.add(Parameter::create("SPE number of updates", *spe_num_updates));
    if (spe_tolerance)
        params.add(Parameter::create("SPE tolerance", *spe_tolerance));
    if (nullspace_shift)
        params.add(Parameter::create("diagonal shift of nullspace", *nullspace_shift));
    if (klle_shift)
        params.add(Parameter::create("KLLE regularizer", *klle_shift));
    if (fa_epsilon)
        params.add(Parameter::create("epsilon of FA", *fa_epsilon));
    if (check_connectivity)
        params.add(Parameter::create("check connectivity", *check_connectivity));

    return with(params).embedUsing(data).embedding;
}

NB_MODULE(tapkee, m) {
    m.doc() = "Tapkee dimensionality reduction library";

    // High-level API
    m.def("embed", &embed,
        nb::arg("data"),
        nb::arg("method") = "lle",
        nb::arg("num_neighbors") = nb::none(),
        nb::arg("target_dimension") = nb::none(),
        nb::arg("gaussian_kernel_width") = nb::none(),
        nb::arg("landmark_ratio") = nb::none(),
        nb::arg("max_iteration") = nb::none(),
        nb::arg("diffusion_map_timesteps") = nb::none(),
        nb::arg("sne_perplexity") = nb::none(),
        nb::arg("sne_theta") = nb::none(),
        nb::arg("squishing_rate") = nb::none(),
        nb::arg("spe_global_strategy") = nb::none(),
        nb::arg("spe_num_updates") = nb::none(),
        nb::arg("spe_tolerance") = nb::none(),
        nb::arg("nullspace_shift") = nb::none(),
        nb::arg("klle_shift") = nb::none(),
        nb::arg("fa_epsilon") = nb::none(),
        nb::arg("check_connectivity") = nb::none(),
        R"doc(
        Embed data using dimensionality reduction.

        Args:
            data: Input matrix (features x samples)
            method: Reduction method ('lle', 'isomap', 'pca', 't-sne', etc.)
            num_neighbors: Number of neighbors for local methods (default: 5)
            target_dimension: Output dimensionality (default: 2)
            gaussian_kernel_width: Kernel width for Laplacian Eigenmaps, DM, LPP
            landmark_ratio: Ratio of landmarks for l-isomap, l-mds (0-1)
            max_iteration: Max iterations for iterative methods
            diffusion_map_timesteps: Timesteps for Diffusion Map
            sne_perplexity: Perplexity for t-SNE (default: 30)
            sne_theta: Theta for Barnes-Hut t-SNE (default: 0.5)
            squishing_rate: Rate for Manifold Sculpting
            spe_global_strategy: Use global SPE strategy
            spe_num_updates: Number of SPE updates
            spe_tolerance: SPE tolerance
            nullspace_shift: Regularizer for eigenproblems
            klle_shift: KLLE regularizer
            fa_epsilon: Factor Analysis epsilon
            check_connectivity: Check graph connectivity

        Returns:
            Embedding matrix (samples x target_dimension)
        )doc"
    );

    // Low-level API (kept for backwards compatibility)
    m.def("withParameters", &with);

    nb::class_<ParametersSet>(m, "ParametersSet")
        .def(nb::init<>())
        .def("add", &ParametersSet::add);

    nb::class_<ParametersInitializedState>(m, "ParametersInitializedState")
        .def(nb::init<const ParametersSet&>())
        .def("embedUsing", &ParametersInitializedState::embedUsing);

    nb::class_<TapkeeOutput>(m, "TapkeeOutput")
        .def_rw("embedding", &TapkeeOutput::embedding);

    m.def("parse_reduction_method", &parse_reduction_method);

    nb::class_<DimensionReductionMethod>(m, "DimensionReductionMethod")
        .def_rw("name", &DimensionReductionMethod::name_);

    nb::class_<Parameter>(m, "Parameter")
        .def_static("create", &Parameter::create<DimensionReductionMethod>)
        .def_static("create", &Parameter::create<int>)
        .def_static("create", &Parameter::create<double>)
        .def_static("create", &Parameter::create<bool>);
}
