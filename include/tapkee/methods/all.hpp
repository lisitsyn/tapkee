/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2024 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/methods/base.hpp>
#include <tapkee/routines/diffusion_maps.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/fa.hpp>
#include <tapkee/routines/generalized_eigendecomposition.hpp>
#include <tapkee/routines/isomap.hpp>
#include <tapkee/routines/laplacian_eigenmaps.hpp>
#include <tapkee/routines/locally_linear.hpp>
#include <tapkee/routines/manifold_sculpting.hpp>
#include <tapkee/routines/multidimensional_scaling.hpp>
#include <tapkee/routines/pca.hpp>
#include <tapkee/routines/random_projection.hpp>
#include <tapkee/routines/spe.hpp>
#include <tapkee/external/barnes_hut_sne/tsne.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

#define IMPLEMENTATION(Method)                                                                                                              \
    template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>                             \
    class Method ## Implementation : public ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>    \
    {                                                                                                                                       \
    public:                                                                                                                                 \
        Method ## Implementation(const ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>& other) : \
            ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>(other)                               \
        {                                                                                                                                     \
        }
#define END_IMPLEMENTATION() };

IMPLEMENTATION(KernelLocallyLinearEmbedding)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix =
            linear_weight_matrix(this->begin, this->end, neighbors, this->kernel, this->parameters[nullspace_shift], this->parameters[klle_shift]);
        DenseMatrix embedding = eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                                    SmallestEigenvalues, weight_matrix, this->parameters[target_dimension]).first;

        return TapkeeOutput(embedding, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(KernelLocalTangentSpaceAlignment)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix = tangent_weight_matrix(
            this->begin, this->end, neighbors, this->kernel, this->parameters[target_dimension], this->parameters[nullspace_shift]);
        DenseMatrix embedding = eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                                    SmallestEigenvalues, weight_matrix, this->parameters[target_dimension]).first;

        return TapkeeOutput(embedding, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(DiffusionMap)
    TapkeeOutput embed()
    {
        IndexType target_dimension_value = static_cast<IndexType>(this->parameters[target_dimension]);
        Parameter target_dimension_add = Parameter::create("target_dimension", target_dimension_value + 1);
        DenseSymmetricMatrix diffusion_matrix =
            compute_diffusion_matrix(this->begin, this->end, this->distance, this->parameters[gaussian_kernel_width]);
        EigendecompositionResult decomposition_result =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                               diffusion_matrix, target_dimension_add);
        DenseMatrix embedding = (decomposition_result.first).leftCols(target_dimension_value);
        // scaling with lambda_i^t
        for (IndexType i = 0; i < target_dimension_value; i++)
            embedding.col(i).array() *=
                pow(decomposition_result.second(i), static_cast<IndexType>(this->parameters[diffusion_map_timesteps]));
        // scaling by eigenvector to largest eigenvalue 1
        for (IndexType i = 0; i < target_dimension_value; i++)
            embedding.col(i).array() /= decomposition_result.first.col(target_dimension_value).array();
        return TapkeeOutput(embedding, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(MultidimensionalScaling)
    TapkeeOutput embed()
    {
        DenseSymmetricMatrix distance_matrix = compute_distance_matrix(this->begin, this->end, this->distance);
        centerMatrix(distance_matrix);
        distance_matrix.array() *= -0.5;
        EigendecompositionResult embedding =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                               distance_matrix, this->parameters[target_dimension]);

        for (IndexType i = 0; i < static_cast<IndexType>(this->parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));
        return TapkeeOutput(embedding.first, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(LandmarkMultidimensionalScaling)
    TapkeeOutput embed()
    {
        this->parameters[landmark_ratio].checked().satisfies(InClosedRange<ScalarType>(3.0 / this->n_vectors, 1.0)).orThrow();

        Landmarks landmarks = select_landmarks_random(this->begin, this->end, this->parameters[landmark_ratio]);
        DenseSymmetricMatrix distance_matrix = compute_distance_matrix(this->begin, this->end, landmarks, this->distance);
        DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
        centerMatrix(distance_matrix);
        distance_matrix.array() *= -0.5;
        EigendecompositionResult landmarks_embedding =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                               distance_matrix, this->parameters[target_dimension]);
        for (IndexType i = 0; i < static_cast<IndexType>(this->parameters[target_dimension]); i++)
            landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
        return TapkeeOutput(triangulate(this->begin, this->end, this->distance, landmarks, landmark_distances_squared,
                                        landmarks_embedding, this->parameters[target_dimension]),
                            this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(Isomap)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);
        DenseSymmetricMatrix shortest_distances_matrix =
            compute_shortest_distances_matrix(this->begin, this->end, neighbors, this->distance);
        shortest_distances_matrix = shortest_distances_matrix.array().square();
        centerMatrix(shortest_distances_matrix);
        shortest_distances_matrix.array() *= -0.5;

        EigendecompositionResult embedding =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                                shortest_distances_matrix, this->parameters[target_dimension]);

        for (IndexType i = 0; i < static_cast<IndexType>(this->parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));

        return TapkeeOutput(embedding.first, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(LandmarkIsomap)
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
        return TapkeeOutput(embedding, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(NeighborhoodPreservingEmbedding)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix =
            linear_weight_matrix(this->begin, this->end, neighbors, this->kernel, this->parameters[nullspace_shift], this->parameters[klle_shift]);
        DenseSymmetricMatrixPair eig_matrices =
            construct_neighborhood_preserving_eigenproblem(weight_matrix, this->begin, this->end, this->features, this->current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            this->parameters[eigen_method], this->parameters[computation_strategy], SmallestEigenvalues, eig_matrices.first,
            eig_matrices.second, this->parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
END_IMPLEMENTATION()

IMPLEMENTATION(HessianLocallyLinearEmbedding)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix =
            hessian_weight_matrix(this->begin, this->end, neighbors, this->kernel, this->parameters[target_dimension]);
        return TapkeeOutput(eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                                SmallestEigenvalues, weight_matrix, this->parameters[target_dimension]).first,
                            this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(LaplacianEigenmaps)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);
        Laplacian laplacian = compute_laplacian(this->begin, this->end, neighbors, this->distance, this->parameters[gaussian_kernel_width]);
        return TapkeeOutput(generalized_eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy],
                                                           SmallestEigenvalues, laplacian.first, laplacian.second,
                                                           this->parameters[target_dimension]).first,
                            this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(LocalityPreservingProjections)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);
        Laplacian laplacian = compute_laplacian(this->begin, this->end, neighbors, this->distance, this->parameters[gaussian_kernel_width]);
        DenseSymmetricMatrixPair eigenproblem_matrices = construct_locality_preserving_eigenproblem(
            laplacian.first, laplacian.second, this->begin, this->end, this->features, this->current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            this->parameters[eigen_method], this->parameters[computation_strategy], SmallestEigenvalues,
            eigenproblem_matrices.first, eigenproblem_matrices.second, this->parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
END_IMPLEMENTATION()

IMPLEMENTATION(PCA)
    TapkeeOutput embed()
    {
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        DenseSymmetricMatrix centered_covariance_matrix =
            compute_covariance_matrix(this->begin, this->end, mean_vector, this->features, this->current_dimension);
        EigendecompositionResult projection_result =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                               centered_covariance_matrix, this->parameters[target_dimension]);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
END_IMPLEMENTATION()

IMPLEMENTATION(RandomProjection)
    TapkeeOutput embed()
    {
        DenseMatrix projection_matrix = gaussian_projection_matrix(this->current_dimension, this->parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);

        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_matrix, mean_vector));
        return TapkeeOutput(project(projection_matrix, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
END_IMPLEMENTATION()

IMPLEMENTATION(KernelPCA)
    TapkeeOutput embed()
    {
        DenseSymmetricMatrix centered_kernel_matrix = compute_centered_kernel_matrix(this->begin, this->end, this->kernel);
        EigendecompositionResult embedding =
            eigendecomposition(this->parameters[eigen_method], this->parameters[computation_strategy], LargestEigenvalues,
                               centered_kernel_matrix, this->parameters[target_dimension]);
        for (IndexType i = 0; i < static_cast<IndexType>(this->parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));
        return TapkeeOutput(embedding.first, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(LinearLocalTangentSpaceAlignment)
    TapkeeOutput embed()
    {
        Neighbors neighbors = this->findNeighborsWith(this->kernel_distance);
        SparseWeightMatrix weight_matrix = tangent_weight_matrix(
            this->begin, this->end, neighbors, this->kernel, this->parameters[target_dimension], this->parameters[nullspace_shift]);
        DenseSymmetricMatrixPair eig_matrices =
            construct_lltsa_eigenproblem(weight_matrix, this->begin, this->end, this->features, this->current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            this->parameters[eigen_method], this->parameters[computation_strategy], SmallestEigenvalues, eig_matrices.first,
            eig_matrices.second, this->parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, this->begin, this->end, this->features, this->current_dimension),
                            projecting_function);
    }
END_IMPLEMENTATION()

IMPLEMENTATION(StochasticProximityEmbedding)
    TapkeeOutput embed()
    {
        Neighbors neighbors;
        if (this->parameters[spe_global_strategy].is(false))
        {
            neighbors = this->findNeighborsWith(this->plain_distance);
        }

        return TapkeeOutput(spe_embedding(this->begin, this->end, this->distance, neighbors, this->parameters[target_dimension],
                                          this->parameters[spe_global_strategy], this->parameters[spe_tolerance],
                                          this->parameters[spe_num_updates], this->parameters[max_iteration]),
                            this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(PassThru)
    TapkeeOutput embed()
    {
        DenseMatrix feature_matrix = dense_matrix_from_features(this->features, this->current_dimension, this->begin, this->end);
        return TapkeeOutput(feature_matrix.transpose(), this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(FactorAnalysis)
    TapkeeOutput embed()
    {
        DenseVector mean_vector = compute_mean(this->begin, this->end, this->features, this->current_dimension);
        return TapkeeOutput(project(this->begin, this->end, this->features, this->current_dimension, this->parameters[max_iteration],
                                    this->parameters[fa_epsilon], this->parameters[target_dimension], mean_vector),
                            this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(tDistributedStochasticNeighborEmbedding)
    TapkeeOutput embed()
    {
        this->parameters[sne_perplexity].checked().satisfies(InClosedRange<ScalarType>(0.0, (this->n_vectors - 1) / 3.0)).orThrow();

        DenseMatrix data = dense_matrix_from_features(this->features, this->current_dimension, this->begin, this->end);

        DenseMatrix embedding(static_cast<IndexType>(this->parameters[target_dimension]), this->n_vectors);
        tsne::TSNE tsne;
        tsne.run(data, data.cols(), data.rows(), embedding.data(), this->parameters[target_dimension],
                 this->parameters[sne_perplexity], this->parameters[sne_theta]);

        return TapkeeOutput(embedding.transpose(), this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

IMPLEMENTATION(ManifoldSculpting)
    TapkeeOutput embed()
    {
        this->parameters[squishing_rate].checked().satisfies(InRange<ScalarType>(0.0, 1.0)).orThrow();

        DenseMatrix embedding = dense_matrix_from_features(this->features, this->current_dimension, this->begin, this->end);

        Neighbors neighbors = this->findNeighborsWith(this->plain_distance);

        manifold_sculpting_embed(this->begin, this->end, embedding, this->parameters[target_dimension], neighbors, this->distance,
                                 this->parameters[max_iteration], this->parameters[squishing_rate]);

        return TapkeeOutput(embedding, this->unimplementedProjectingFunction());
    }
END_IMPLEMENTATION()

#undef IMPLEMENTATION
#undef END_IMPLEMENTATION

} // End of namespace tapkee_internal
} // End of namespace tapkee
