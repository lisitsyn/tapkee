/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/pca.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(KernelPrincipalComponentAnalysis)
    TapkeeOutput embed()
    {
        DenseSymmetricMatrix centered_kernel_matrix = compute_centered_kernel_matrix(begin, end, kernel);
        EigendecompositionResult embedding =
            eigendecomposition(parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues,
                               centered_kernel_matrix, parameters[target_dimension]);
        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));
        return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
