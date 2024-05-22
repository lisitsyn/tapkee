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
    void validate()
    {
        parameters[gaussian_kernel_width].checked().satisfies(Positivity<ScalarType>()).orThrow();
    }

    TapkeeOutput embed()
    {
        Neighbors neighbors = find_neighbors_with(plain_distance);
        Laplacian laplacian = compute_laplacian(begin, end, neighbors, distance, parameters[gaussian_kernel_width]);
        return TapkeeOutput(generalized_eigendecomposition(parameters[eigen_method], parameters[computation_strategy],
                                                           SmallestEigenvalues, laplacian.first, laplacian.second,
                                                           parameters[target_dimension]).first,
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
