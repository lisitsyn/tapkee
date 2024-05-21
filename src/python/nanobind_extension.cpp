#include <cli/util.hpp>                // for parse_reduction_method, TODO consider moving from src to include

#include <stichwort/parameter.hpp>     // for Parameter::create

#include <tapkee/chain_interface.hpp>  // for initialize, withParameters
#include <tapkee/defines.hpp>          // for ParametersSet, TapkeeOutput
#include <tapkee/defines/methods.hpp>  // for DimensionReductionMethod

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>      // for type caster embedUsing
#include <nanobind/stl/string.h>       // for type caster calling stichwort::Parameter::create(const std::string&, ...

namespace nb = nanobind;

using stichwort::Parameter;

using tapkee::initialize;
using tapkee::DimensionReductionMethod;
using tapkee::ParametersSet;
using tapkee::TapkeeOutput;

using tapkee::tapkee_internal::ParametersInitializedState;  // TODO consider making it part of the "external" API

NB_MODULE(tapkee, m) {
    nb::class_<initialize>(m, "initialize")
        .def(nb::init<>())
        .def("withParameters", &initialize::withParameters);

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
        .def_static("create", &Parameter::create<int>);
}
