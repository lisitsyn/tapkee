/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2024 Sergey Lisitsyn
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

template <class RandomAccessIterator>
Landmarks select_landmarks_random(RandomAccessIterator begin, RandomAccessIterator end, ScalarType ratio)
{
    Landmarks landmarks;
    landmarks.reserve(end - begin);
    for (RandomAccessIterator iter = begin; iter != end; ++iter)
        landmarks.push_back(iter - begin);
    tapkee::random_shuffle(landmarks.begin(), landmarks.end());
    landmarks.erase(landmarks.begin() + static_cast<IndexType>(landmarks.size() * ratio), landmarks.end());
    return landmarks;
}

template <class RandomAccessIterator, class PairwiseCallback>
DenseMatrix triangulate(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback distance_callback,
                        Landmarks& landmarks, DenseVector& landmark_distances_squared,
                        EigendecompositionResult& landmarks_embedding, IndexType target_dimension)
{
    timed_context context("Landmark triangulation");

    const IndexType n_vectors = end - begin;
    const IndexType n_landmarks = landmarks.size();

    std::vector<bool> to_process(n_vectors, true);

    DenseMatrix embedding(n_vectors, target_dimension);

    for (IndexType index_iter = 0; index_iter < n_landmarks; ++index_iter)
    {
        to_process[landmarks[index_iter]] = false;
        embedding.row(landmarks[index_iter]).noalias() = landmarks_embedding.first.row(index_iter);
    }

    for (IndexType i = 0; i < target_dimension; ++i)
        landmarks_embedding.first.col(i).array() /= landmarks_embedding.second(i);

#pragma omp parallel
    {
        DenseVector distances_to_landmarks(n_landmarks);
        IndexType index_iter;
#pragma omp for nowait
        for (index_iter = 0; index_iter < n_vectors; ++index_iter)
        {
            if (!to_process[index_iter])
                continue;

            for (IndexType i = 0; i < n_landmarks; ++i)
            {
                ScalarType d = distance_callback.distance(begin[index_iter], begin[landmarks[i]]);
                distances_to_landmarks(i) = d * d;
            }
            // distances_to_landmarks.array().square();

            distances_to_landmarks -= landmark_distances_squared;
            embedding.row(index_iter).noalias() = -0.5 * landmarks_embedding.first.transpose() * distances_to_landmarks;
        }
    }

    return embedding;
}

} // End of namespace tapkee_internal
} // End of namespace tapkee
