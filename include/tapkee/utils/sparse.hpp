/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Vladyslav Gorbatiuk
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

inline SparseMatrix sparse_matrix_from_triplets(const SparseTriplets& sparse_triplets, IndexType m, IndexType n)
{
    SparseMatrix matrix(m, n);
    matrix.setFromTriplets(sparse_triplets.begin(), sparse_triplets.end());
    return matrix;
}

} // namespace tapkee_internal
} // namespace tapkee
