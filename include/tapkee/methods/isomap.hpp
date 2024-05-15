/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/isomap.hpp>
#include <tapkee/routines/multidimensional_scaling.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(Isomap)
    TapkeeOutput embed()
    {
        Neighbors neighbors = findNeighborsWith(plain_distance);
        DenseSymmetricMatrix shortest_distances_matrix =
            compute_shortest_distances_matrix(begin, end, neighbors, distance);
        shortest_distances_matrix = shortest_distances_matrix.array().square();
        centerMatrix(shortest_distances_matrix);
        shortest_distances_matrix.array() *= -0.5;

        EigendecompositionResult embedding =
            eigendecomposition(parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues,
                                shortest_distances_matrix, parameters[target_dimension]);

        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));

        return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
