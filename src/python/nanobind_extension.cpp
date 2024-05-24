#include <cli/util.hpp>

#include <stichwort/parameter.hpp>

#include <tapkee/chain_interface.hpp>
#include <tapkee/defines.hpp>
#include <tapkee/defines/methods.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

using stichwort::Parameter;

using tapkee::initialize;
using tapkee::DimensionReductionMethod;
using tapkee::ParametersSet;
using tapkee::TapkeeOutput;

using tapkee::tapkee_internal::ParametersInitializedState;

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

    m.def("parse_multiple", &parse_multiple);

    nb::class_<DimensionReductionMethod>(m, "DimensionReductionMethod")
        .def_rw("name", &DimensionReductionMethod::name_);

    nb::class_<Parameter>(m, "Parameter")
        .def_static("create", &Parameter::create<DimensionReductionMethod>)
        .def_static("create", &Parameter::create<int>);
}
