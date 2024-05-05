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
    TapkeeOutput embed()
    {
        this->parameters[squishing_rate].checked().satisfies(InRange<ScalarType>(0.0, 1.0)).orThrow();

        DenseMatrix embedding = dense_matrix_from_features(this->features, this->current_dimension, this->begin, this->end);

        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);

        manifold_sculpting_embed(this->begin, this->end, embedding, this->parameters[target_dimension], neighbors, this->distance,
                                 this->parameters[max_iteration], this->parameters[squishing_rate]);

        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
