/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/random_projection.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(RandomProjection)
    void validate()
    {
    }

    TapkeeOutput embed()
    {
        DenseMatrix projection_matrix = gaussian_projection_matrix(current_dimension, parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);

        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_matrix, mean_vector));
        return TapkeeOutput(project(projection_matrix, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
