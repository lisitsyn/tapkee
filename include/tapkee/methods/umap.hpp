/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(UniformManifoldApproximationAndProjection)
    TapkeeOutput embed()
    {
        DenseMatrix feature_matrix{this->end - this->begin, static_cast<IndexType>(this->parameters[target_dimension])};
        return TapkeeOutput(feature_matrix, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
