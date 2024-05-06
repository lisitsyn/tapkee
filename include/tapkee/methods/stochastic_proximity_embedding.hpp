/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/spe.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(StochasticProximityEmbedding)
    TapkeeOutput embed()
    {
        this->parameters[spe_tolerance].checked().satisfies(Positivity<ScalarType>()).orThrow();
        this->parameters[spe_num_updates].checked().satisfies(Positivity<IndexType>()).orThrow();
        Neighbors neighbors;
        if (this->parameters[spe_global_strategy].is(false))
        {
            neighbors = this->findNeighborsWith(this->plain_distance);
        }

        return TapkeeOutput(spe_embedding(this->begin, this->end, this->distance, neighbors, this->parameters[target_dimension],
                                          this->parameters[spe_global_strategy], this->parameters[spe_tolerance],
                                          this->parameters[spe_num_updates], this->parameters[max_iteration]),
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
