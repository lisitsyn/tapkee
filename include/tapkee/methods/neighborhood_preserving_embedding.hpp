/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/generalized_eigendecomposition.hpp>
#include <tapkee/routines/locally_linear.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(NeighborhoodPreservingEmbedding)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix =
            linear_weight_matrix(this->begin, this->end, neighbors, this->kernel, this->parameters[nullspace_shift], this->parameters[klle_shift]);
        DenseSymmetricMatrixPair eig_matrices =
            construct_neighborhood_preserving_eigenproblem(weight_matrix, this->begin, this->end, this->features, this->current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            this->parameters[eigen_method], this->parameters[computation_strategy], SmallestEigenvalues, eig_matrices.first,
            eig_matrices.second, this->parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
