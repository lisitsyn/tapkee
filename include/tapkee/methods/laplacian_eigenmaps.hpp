/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/laplacian_eigenmaps.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(LaplacianEigenmaps)
    TapkeeOutput embed()
    {
        this->parameters[gaussian_kernel_width].checked().satisfies(Positivity<ScalarType>()).orThrow();

        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);
        Laplacian laplacian = compute_laplacian(this->begin, this->end, neighbors, this->distance, this->parameters[gaussian_kernel_width]);
        return TapkeeOutput(generalized_eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                                           SmallestEigenvalues, laplacian.first, laplacian.second,
                                                           this->parameters[target_dimension]).first,
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
