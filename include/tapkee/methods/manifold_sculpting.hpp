/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/manifold_sculpting.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(ManifoldSculpting)
    void validate()
    {
        parameters[squishing_rate].checked().satisfies(InRange<ScalarType>(0.0, 1.0)).orThrow();
    }

    TapkeeOutput embed()
    {

        DenseMatrix embedding = dense_matrix_from_features(features, current_dimension, begin, end);

        Neighbors neighbors = find_neighbors_with(plain_distance);

        manifold_sculpting_embed(begin, end, embedding, parameters[target_dimension], neighbors, distance,
                                 parameters[max_iteration], parameters[squishing_rate]);

        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
