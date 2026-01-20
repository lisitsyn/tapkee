/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines/stdtypes.hpp>
#include <tapkee/defines/types.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

typedef Eigen::Triplet<tapkee::ScalarType> SparseTriplet;

typedef TAPKEE_INTERNAL_VECTOR<tapkee::tapkee_internal::SparseTriplet> SparseTriplets;
typedef TAPKEE_INTERNAL_VECTOR<tapkee::IndexType> LocalNeighbors;
typedef TAPKEE_INTERNAL_VECTOR<tapkee::tapkee_internal::LocalNeighbors> Neighbors;
typedef TAPKEE_INTERNAL_PAIR<tapkee::DenseMatrix, tapkee::DenseVector> EigendecompositionResult;
typedef TAPKEE_INTERNAL_VECTOR<tapkee::IndexType> Landmarks;
typedef TAPKEE_INTERNAL_PAIR<tapkee::SparseWeightMatrix, tapkee::DenseDiagonalMatrix> Laplacian;
typedef TAPKEE_INTERNAL_PAIR<tapkee::DenseSymmetricMatrix, tapkee::DenseSymmetricMatrix> DenseSymmetricMatrixPair;
typedef TAPKEE_INTERNAL_PAIR<tapkee::SparseMatrix, tapkee::tapkee_internal::Neighbors> SparseMatrixNeighborsPair;

#if defined(TAPKEE_USE_PRIORITY_QUEUE) && defined(TAPKEE_USE_FIBONACCI_HEAP)
#error "Can't use both priority queue and fibonacci heap at the same time"
#endif
#if !defined(TAPKEE_USE_PRIORITY_QUEUE) && !defined(TAPKEE_USE_FIBONACCI_HEAP)
#define TAPKEE_USE_PRIORITY_QUEUE
#endif

} // End of namespace tapkee_internal

} // namespace tapkee
