/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/multidimensional_scaling.hpp>
#include <tapkee/routines/landmarks.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(LandmarkMultidimensionalScaling)
    TapkeeOutput embed()
    {
        parameters[landmark_ratio].checked().satisfies(InClosedRange<ScalarType>(3.0 / n_vectors, 1.0)).orThrow();

        Landmarks landmarks = select_landmarks_random(begin, end, parameters[landmark_ratio]);
        DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin, end, landmarks, distance);
        DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
        centerMatrix(distance_matrix);
        distance_matrix.array() *= -0.5;
        EigendecompositionResult landmarks_embedding =
            eigendecomposition_via(LargestEigenvalues, distance_matrix, parameters[target_dimension]);
        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
        return TapkeeOutput(triangulate(begin, end, distance, landmarks, landmark_distances_squared,
                                        landmarks_embedding, parameters[target_dimension]),
                            unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
