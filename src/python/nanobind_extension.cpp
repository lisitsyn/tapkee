#include <tapkee/chain_interface.hpp>  // for initialize, withParameters
#include <tapkee/defines.hpp>          // for ParametersSet, TapkeeOutput
#include <tapkee/defines/types.hpp>    // for DenseMatrix

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

using tapkee::initialize;
// using tapkee::DenseMatrix;  // superseded by typecaster in nanobind/eigen/dense.h
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
}
