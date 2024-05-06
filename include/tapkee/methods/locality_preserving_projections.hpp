/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/diffusion_maps.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/generalized_eigendecomposition.hpp>
#include <tapkee/routines/laplacian_eigenmaps.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(LocalityPreservingProjections)
    TapkeeOutput embed()
    {
        this->parameters[gaussian_kernel_width].checked().satisfies(Positivity<ScalarType>()).orThrow();

        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);
        Laplacian laplacian = compute_laplacian(this->begin, this->end, neighbors, this->distance, this->parameters[gaussian_kernel_width]);
        DenseSymmetricMatrixPair eigenproblem_matrices = construct_locality_preserving_eigenproblem(
            laplacian.first, laplacian.second, this->begin, this->end, this->features, this->current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            this->parameters[eigen_method], this->parameters[computation_strategy], SmallestEigenvalues,
            eigenproblem_matrices.first, eigenproblem_matrices.second, this->parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
