/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/pca.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(PrincipalComponentAnalysis)
    TapkeeOutput embed()
    {
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        DenseSymmetricMatrix centered_covariance_matrix =
            compute_covariance_matrix(begin, end, mean_vector, features, current_dimension);
        EigendecompositionResult projection_result =
            eigendecomposition(parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues,
                               centered_covariance_matrix, parameters[target_dimension]);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
