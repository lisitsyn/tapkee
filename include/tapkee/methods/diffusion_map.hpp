/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/diffusion_maps.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(DiffusionMap)
    TapkeeOutput embed()
    {
        parameters[diffusion_map_timesteps].checked().satisfies(Positivity<IndexType>()).orThrow();
        parameters[gaussian_kernel_width].checked().satisfies(Positivity<ScalarType>()).orThrow();

        IndexType target_dimension_value = static_cast<IndexType>(parameters[target_dimension]);
        Parameter target_dimension_add = Parameter::create("target_dimension", target_dimension_value + 1);
        DenseSymmetricMatrix diffusion_matrix =
            compute_diffusion_matrix(begin, end, distance, parameters[gaussian_kernel_width]);
        EigendecompositionResult decomposition_result =
            eigendecomposition(parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues,
                               diffusion_matrix, target_dimension_add);
        DenseMatrix embedding = (decomposition_result.first).leftCols(target_dimension_value);
        // scaling with lambda_i^t
        for (IndexType i = 0; i < target_dimension_value; i++)
            embedding.col(i).array() *=
                pow(decomposition_result.second(i), static_cast<IndexType>(parameters[diffusion_map_timesteps]));
        // scaling by eigenvector to largest eigenvalue 1
        for (IndexType i = 0; i < target_dimension_value; i++)
            embedding.col(i).array() /= decomposition_result.first.col(target_dimension_value).array();
        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
