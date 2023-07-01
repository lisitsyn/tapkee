/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

inline DenseMatrix gaussian_projection_matrix(IndexType target_dimension, IndexType current_dimension)
{
    DenseMatrix projection_matrix(target_dimension, current_dimension);

    for (IndexType i = 0; i < target_dimension; ++i)
    {
        for (IndexType j = 0; j < current_dimension; ++j)
        {
            projection_matrix(i, j) = tapkee::gaussian_random() / sqrt(static_cast<ScalarType>(target_dimension));
        }
    }

    return projection_matrix;
}

} // namespace tapkee_internal
} // namespace tapkee
