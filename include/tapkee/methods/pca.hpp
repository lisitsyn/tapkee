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

__TAPKEE_IMPLEMENTATION(PCA)
    TapkeeOutput embed()
    {
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        DenseSymmetricMatrix centered_covariance_matrix =
            compute_covariance_matrix(this->begin, this->end, mean_vector, this->features, this->current_dimension);
        EigendecompositionResult projection_result =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                               centered_covariance_matrix, this->parameters[target_dimension]);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
