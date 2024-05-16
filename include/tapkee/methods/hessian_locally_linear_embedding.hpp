/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/locally_linear.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(HessianLocallyLinearEmbedding)
    TapkeeOutput embed()
    {
        Neighbors neighbors = find_neighbors_with(kernel_distance);
        SparseWeightMatrix weight_matrix =
            hessian_weight_matrix(begin, end, neighbors, kernel, parameters[target_dimension]);
        return TapkeeOutput(eigendecomposition_via(SmallestEigenvalues, weight_matrix, parameters[target_dimension]).first,
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
