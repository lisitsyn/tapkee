/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

namespace tapkee
{
namespace tapkee_internal
{

inline void centerMatrix(DenseMatrix& matrix)
{
    DenseVector col_means = matrix.colwise().mean().transpose();
    DenseMatrix::Scalar grand_mean = matrix.mean();
    matrix.array() += grand_mean;
    matrix.rowwise() -= col_means.transpose();
    matrix.colwise() -= col_means;
}

} // namespace tapkee_internal
} // namespace tapkee
