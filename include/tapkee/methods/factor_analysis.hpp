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
    void validate()
    {
        parameters[fa_epsilon].checked().satisfies(NonNegativity<ScalarType>()).orThrow();
    }

    TapkeeOutput embed()
    {
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        return TapkeeOutput(project(begin, end, features, current_dimension, parameters[max_iteration],
                                    parameters[fa_epsilon], parameters[target_dimension], mean_vector),
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
