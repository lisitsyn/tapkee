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

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_distance_matrix(RandomAccessIterator begin, RandomAccessIterator /*end*/,
                                             Landmarks& landmarks, PairwiseCallback callback)
{
    timed_context context("Multidimensional scaling distance matrix computation");

    const IndexType n_landmarks = landmarks.size();
    DenseSymmetricMatrix distance_matrix(n_landmarks, n_landmarks);

#pragma omp parallel
    {
        IndexType i_index_iter, j_index_iter;
#pragma omp for nowait
        for (i_index_iter = 0; i_index_iter < n_landmarks; ++i_index_iter)
        {
            for (j_index_iter = i_index_iter; j_index_iter < n_landmarks; ++j_index_iter)
            {
                ScalarType d = callback.distance(begin[landmarks[i_index_iter]], begin[landmarks[j_index_iter]]);
                d *= d;
                distance_matrix(i_index_iter, j_index_iter) = d;
                distance_matrix(j_index_iter, i_index_iter) = d;
            }
        }
    }
    return distance_matrix;
}

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_distance_matrix(RandomAccessIterator begin, RandomAccessIterator end,
                                             PairwiseCallback callback)
{
    timed_context context("Multidimensional scaling distance matrix computation");

    const IndexType n_vectors = end - begin;
    DenseSymmetricMatrix distance_matrix(n_vectors, n_vectors);

#pragma omp parallel
    {
        IndexType i_index_iter, j_index_iter;
#pragma omp for nowait
        for (i_index_iter = 0; i_index_iter < n_vectors; ++i_index_iter)
        {
            for (j_index_iter = i_index_iter; j_index_iter < n_vectors; ++j_index_iter)
            {
                ScalarType d = callback.distance(begin[i_index_iter], begin[j_index_iter]);
                d *= d;
                distance_matrix(i_index_iter, j_index_iter) = d;
                distance_matrix(j_index_iter, i_index_iter) = d;
            }
        }
    }
    return distance_matrix;
}

} // End of namespace tapkee_internal
} // End of namespace tapkee
