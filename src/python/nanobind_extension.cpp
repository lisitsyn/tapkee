#include <string>

#include <stichwort/parameter.hpp>     // for Parameter::create

#include <tapkee/chain_interface.hpp>  // for initialize, withParameters
#include <tapkee/defines.hpp>          // for ParametersSet, TapkeeOutput
#include <tapkee/defines/methods.hpp>  // for DimensionReductionMethod
#include <tapkee/defines/types.hpp>    // for DenseMatrix

#include <cli/util.hpp> // TODO consider moving to include

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

using stichwort::Parameter;

using tapkee::initialize;
// using tapkee::DenseMatrix;  // superseded by typecaster in nanobind/eigen/dense.h
using tapkee::DimensionReductionMethod;
using tapkee::ParametersSet;
using tapkee::TapkeeOutput;

using tapkee::tapkee_internal::ParametersInitializedState;  // externalize in tapkee?

NB_MODULE(tapkee, m) {
    nb::class_<initialize>(m, "initialize")
        .def(nb::init<>())
        .def("withParameters", &initialize::withParameters);

    nb::class_<ParametersSet>(m, "ParametersSet")
        .def(nb::init<>());

    nb::class_<ParametersInitializedState>(m, "ParametersInitializedState")
        .def(nb::init<const ParametersSet&>())
        // TapkeeOutput embedUsing(const DenseMatrix& matrix) const
        .def("embedUsing", &ParametersInitializedState::embedUsing);

    //nb::class_<DenseMatrix>(m, "DenseMatrix");  // compilation error together w/ type caster
    nb::class_<TapkeeOutput>(m, "TapkeeOutput");

    m.def("parse_reduction_method", &parse_reduction_method);

    nb::class_<DimensionReductionMethod>(m, "DimensionReductionMethod")
        .def_rw("name", &DimensionReductionMethod::name_);

    nb::class_<Parameter>(m, "Parameter")
        .def("create", &Parameter::create<int>)
        .def("create", &Parameter::create<std::string>);
}
