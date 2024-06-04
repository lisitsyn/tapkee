/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/generalized_eigendecomposition.hpp>
#include <tapkee/routines/locally_linear.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(LinearLocalTangentSpaceAlignment)
    void validate()
    {
    }

    TapkeeOutput embed()
    {
        Neighbors neighbors = find_neighbors_with(kernel_distance);
        SparseWeightMatrix weight_matrix = tangent_weight_matrix(
            begin, end, neighbors, kernel, parameters[target_dimension], parameters[nullspace_shift]);
        DenseSymmetricMatrixPair eig_matrices =
            construct_lltsa_eigenproblem(weight_matrix, begin, end, features, current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues, eig_matrices.first,
            eig_matrices.second, parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
