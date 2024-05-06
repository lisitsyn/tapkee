/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/fa.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(FactorAnalysis)
    TapkeeOutput embed()
    {
        this->parameters[fa_epsilon].checked().satisfies(NonNegativity<ScalarType>()).orThrow();

        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        return TapkeeOutput(project(this->begin, this->end, this->features, this->current_dimension, this->parameters[max_iteration],
                                    this->parameters[fa_epsilon], this->parameters[target_dimension], mean_vector),
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
