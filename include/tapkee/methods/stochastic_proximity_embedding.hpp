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
        parameters[spe_tolerance].checked().satisfies(Positivity<ScalarType>()).orThrow();
        parameters[spe_num_updates].checked().satisfies(Positivity<IndexType>()).orThrow();
        Neighbors neighbors;
        if (parameters[spe_global_strategy].is(false))
        {
            neighbors = findNeighborsWith(plain_distance);
        }

        return TapkeeOutput(spe_embedding(begin, end, distance, neighbors, parameters[target_dimension],
                                          parameters[spe_global_strategy], parameters[spe_tolerance],
                                          parameters[spe_num_updates], parameters[max_iteration]),
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
