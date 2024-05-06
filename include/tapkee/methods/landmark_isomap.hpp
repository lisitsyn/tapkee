/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/isomap.hpp>
#include <tapkee/routines/multidimensional_scaling.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

__TAPKEE_IMPLEMENTATION(LandmarkIsomap)
    TapkeeOutput embed()
    {
        this->parameters[landmark_ratio].checked().satisfies(InClosedRange<ScalarType>(3.0 / this->n_vectors, 1.0)).orThrow();

        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);
        Landmarks landmarks = select_landmarks_random(this->begin, this->end, this->parameters[landmark_ratio]);
        DenseMatrix distance_matrix = compute_shortest_distances_matrix(this->begin, this->end, landmarks, neighbors, this->distance);
        distance_matrix = distance_matrix.array().square();

        DenseVector col_means = distance_matrix.colwise().mean();
        DenseVector row_means = distance_matrix.rowwise().mean();
        ScalarType grand_mean = distance_matrix.mean();
        distance_matrix.array() += grand_mean;
        distance_matrix.colwise() -= row_means;
        distance_matrix.rowwise() -= col_means.transpose();
        distance_matrix.array() *= -0.5;

        EigendecompositionResult landmarks_embedding;

        if (this->parameters[eigen_method].is(Dense))
        {
            DenseMatrix distance_matrix_sym = distance_matrix * distance_matrix.transpose();
            landmarks_embedding =
                eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                                   distance_matrix_sym, this->parameters[target_dimension]);
        }
        else
        {
            landmarks_embedding =
                eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                   SquaredLargestEigenvalues, distance_matrix, this->parameters[target_dimension]);
        }

        DenseMatrix embedding = distance_matrix.transpose() * landmarks_embedding.first;

        for (IndexType i = 0; i < static_cast<IndexType>(this->parameters[target_dimension]); i++)
            embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }
__TAPKEE_END_IMPLEMENTATION()

} // End of namespace tapkee_internal
} // End of namespace tapkee
