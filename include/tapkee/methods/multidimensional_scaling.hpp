/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/multidimensional_scaling.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(MultidimensionalScaling)
    void validate()
    {
    }

    TapkeeOutput embed()
    {
        DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin, end, distance);
        centerMatrix(distance_matrix);
        distance_matrix.array() *= -0.5;
        EigendecompositionResult embedding =
            eigendecomposition_via(LargestEigenvalues, distance_matrix, parameters[target_dimension]);

        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));
        return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
