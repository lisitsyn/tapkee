/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/locally_linear.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(KernelLocalTangentSpaceAlignment)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix = tangent_weight_matrix(
            this->begin, this->end, neighbors, this->kernel, this->parameters[target_dimension], this->parameters[nullspace_shift]);
        DenseMatrix embedding = eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                                    SmallestEigenvalues, weight_matrix, this->parameters[target_dimension]).first;

        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
