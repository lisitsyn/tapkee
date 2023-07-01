/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/predicates.hpp>
#include <tapkee/external/barnes_hut_sne/tsne.hpp>
#include <tapkee/neighbors/neighbors.hpp>
#include <tapkee/parameters/context.hpp>
#include <tapkee/parameters/defaults.hpp>
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
#include <tapkee/utils/features.hpp>
#include <tapkee/utils/logging.hpp>
#include <tapkee/utils/naming.hpp>
#include <tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{


template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>
class ImplementationBase
{
  public:
    ImplementationBase(RandomAccessIterator b, RandomAccessIterator e, KernelCallback k, DistanceCallback d,
                       FeaturesCallback f, ParametersSet& pmap, const Context& ctx)
        : parameters(pmap), context(ctx), kernel(k), distance(d), features(f),
          plain_distance(PlainDistance<RandomAccessIterator, DistanceCallback>(distance)),
          kernel_distance(KernelDistance<RandomAccessIterator, KernelCallback>(kernel)), begin(b), end(e),
          n_vectors(0), current_dimension(0)
    {
        n_vectors = (end - begin);

        if (n_vectors == 0)
            throw no_data_error();

        validateParameters();

        if (!is_dummy<FeaturesCallback>::value)
            current_dimension = features.dimension();
        else
            current_dimension = 0;
    }

    void validateParameters() const
    {
        parameters[target_dimension].checked().satisfies(InRange<IndexType>(1, n_vectors)).orThrow();
        parameters[gaussian_kernel_width].checked().satisfies(Positivity<ScalarType>()).orThrow();
        parameters[diffusion_map_timesteps].checked().satisfies(Positivity<IndexType>()).orThrow();
        parameters[spe_tolerance].checked().satisfies(Positivity<ScalarType>()).orThrow();
        parameters[spe_num_updates].checked().satisfies(Positivity<IndexType>()).orThrow();
        parameters[sne_theta].checked().satisfies(NonNegativity<ScalarType>()).orThrow();
        parameters[fa_epsilon].checked().satisfies(NonNegativity<ScalarType>()).orThrow();
        parameters[sne_perplexity].checked().satisfies(NonNegativity<ScalarType>()).orThrow();
    }

    TapkeeOutput embedUsing(DimensionReductionMethod method)
    {
        if (context.is_cancelled())
            throw cancelled_exception();

#define tapkee_method_handle(X)                                                                                        \
    case X: {                                                                                                          \
        timed_context tctx__("[+] embedding with " #X);                                                                \
        if (MethodTraits<X>::needs_kernel && is_dummy<KernelCallback>::value)                                          \
        {                                                                                                              \
            throw unsupported_method_error("Kernel callback is missed");                                               \
        }                                                                                                              \
        if (MethodTraits<X>::needs_distance && is_dummy<DistanceCallback>::value)                                      \
        {                                                                                                              \
            throw unsupported_method_error("Distance callback is missed");                                             \
        }                                                                                                              \
        if (MethodTraits<X>::needs_features && is_dummy<FeaturesCallback>::value)                                      \
        {                                                                                                              \
            throw unsupported_method_error("Features callback is missed");                                             \
        }                                                                                                              \
        return ImplementationBase::embed##X();                                                                         \
    }                                                                                                                  \
    break;

        switch (method)
        {
            tapkee_method_handle(KernelLocallyLinearEmbedding);
            tapkee_method_handle(KernelLocalTangentSpaceAlignment);
            tapkee_method_handle(DiffusionMap);
            tapkee_method_handle(MultidimensionalScaling);
            tapkee_method_handle(LandmarkMultidimensionalScaling);
            tapkee_method_handle(Isomap);
            tapkee_method_handle(LandmarkIsomap);
            tapkee_method_handle(NeighborhoodPreservingEmbedding);
            tapkee_method_handle(LinearLocalTangentSpaceAlignment);
            tapkee_method_handle(HessianLocallyLinearEmbedding);
            tapkee_method_handle(LaplacianEigenmaps);
            tapkee_method_handle(LocalityPreservingProjections);
            tapkee_method_handle(PCA);
            tapkee_method_handle(KernelPCA);
            tapkee_method_handle(RandomProjection);
            tapkee_method_handle(StochasticProximityEmbedding);
            tapkee_method_handle(PassThru);
            tapkee_method_handle(FactorAnalysis);
            tapkee_method_handle(tDistributedStochasticNeighborEmbedding);
            tapkee_method_handle(ManifoldSculpting);
        }
#undef tapkee_method_handle
        return TapkeeOutput();
    }

  private:
    ParametersSet parameters;
    Context context;
    KernelCallback kernel;
    DistanceCallback distance;
    FeaturesCallback features;
    PlainDistance<RandomAccessIterator, DistanceCallback> plain_distance;
    KernelDistance<RandomAccessIterator, KernelCallback> kernel_distance;

    RandomAccessIterator begin;
    RandomAccessIterator end;

    IndexType n_vectors;
    IndexType current_dimension;

    template <class Distance> Neighbors findNeighborsWith(Distance d)
    {
        parameters[num_neighbors].checked().satisfies(InRange<IndexType>(3, n_vectors)).orThrow();
        return find_neighbors(parameters[neighbors_method], begin, end, d, parameters[num_neighbors], parameters[check_connectivity]);
    }

    static tapkee::ProjectingFunction unimplementedProjectingFunction()
    {
        return tapkee::ProjectingFunction();
    }

    TapkeeOutput embedKernelLocallyLinearEmbedding()
    {
        Neighbors neighbors = findNeighborsWith(kernel_distance);
        SparseWeightMatrix weight_matrix =
            linear_weight_matrix(begin, end, neighbors, kernel, parameters[nullspace_shift], parameters[klle_shift]);
        DenseMatrix embedding = eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues,
                                                   weight_matrix, parameters[target_dimension])
                                    .first;

        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }

    TapkeeOutput embedKernelLocalTangentSpaceAlignment()
    {
        Neighbors neighbors = findNeighborsWith(kernel_distance);
        SparseWeightMatrix weight_matrix =
            tangent_weight_matrix(begin, end, neighbors, kernel, parameters[target_dimension], parameters[nullspace_shift]);
        DenseMatrix embedding = eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues,
                                                   weight_matrix, parameters[target_dimension])
                                    .first;

        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }

    TapkeeOutput embedDiffusionMap()
    {
        IndexType target_dimension_value = static_cast<IndexType>(parameters[target_dimension]);
        Parameter target_dimension_add = Parameter::create("target_dimension", target_dimension_value + 1);
        DenseSymmetricMatrix diffusion_matrix = compute_diffusion_matrix(begin, end, distance, parameters[gaussian_kernel_width]);
        EigendecompositionResult decomposition_result = eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues, diffusion_matrix, target_dimension_add);
        DenseMatrix embedding = (decomposition_result.first).leftCols(target_dimension_value);
        // scaling with lambda_i^t
        for (IndexType i = 0; i < target_dimension_value; i++)
            embedding.col(i).array() *= pow(decomposition_result.second(i), static_cast<IndexType>(parameters[diffusion_map_timesteps]));
        // scaling by eigenvector to largest eigenvalue 1
        for (IndexType i = 0; i < target_dimension_value; i++)
            embedding.col(i).array() /= decomposition_result.first.col(target_dimension_value).array();
        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }

    TapkeeOutput embedMultidimensionalScaling()
    {
        DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin, end, distance);
        centerMatrix(distance_matrix);
        distance_matrix.array() *= -0.5;
        EigendecompositionResult embedding = eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues, distance_matrix, parameters[target_dimension]);

        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));
        return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
    }

    TapkeeOutput embedLandmarkMultidimensionalScaling()
    {
        parameters[landmark_ratio].checked().satisfies(InClosedRange<ScalarType>(3.0 / n_vectors, 1.0)).orThrow();

        Landmarks landmarks = select_landmarks_random(begin, end, parameters[landmark_ratio]);
        DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin, end, landmarks, distance);
        DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
        centerMatrix(distance_matrix);
        distance_matrix.array() *= -0.5;
        EigendecompositionResult landmarks_embedding = eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues, distance_matrix, parameters[target_dimension]);
        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
        return TapkeeOutput(triangulate(begin, end, distance, landmarks, landmark_distances_squared,
                                        landmarks_embedding, parameters[target_dimension]),
                            unimplementedProjectingFunction());
    }

    TapkeeOutput embedIsomap()
    {
        Neighbors neighbors = findNeighborsWith(plain_distance);
        DenseSymmetricMatrix shortest_distances_matrix =
            compute_shortest_distances_matrix(begin, end, neighbors, distance);
        shortest_distances_matrix = shortest_distances_matrix.array().square();
        centerMatrix(shortest_distances_matrix);
        shortest_distances_matrix.array() *= -0.5;

        EigendecompositionResult embedding = eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues, shortest_distances_matrix, parameters[target_dimension]);

        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));

        return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
    }

    TapkeeOutput embedLandmarkIsomap()
    {
        parameters[landmark_ratio].checked().satisfies(InClosedRange<ScalarType>(3.0 / n_vectors, 1.0)).orThrow();

        Neighbors neighbors = findNeighborsWith(plain_distance);
        Landmarks landmarks = select_landmarks_random(begin, end, parameters[landmark_ratio]);
        DenseMatrix distance_matrix = compute_shortest_distances_matrix(begin, end, landmarks, neighbors, distance);
        distance_matrix = distance_matrix.array().square();

        DenseVector col_means = distance_matrix.colwise().mean();
        DenseVector row_means = distance_matrix.rowwise().mean();
        ScalarType grand_mean = distance_matrix.mean();
        distance_matrix.array() += grand_mean;
        distance_matrix.colwise() -= row_means;
        distance_matrix.rowwise() -= col_means.transpose();
        distance_matrix.array() *= -0.5;

        EigendecompositionResult landmarks_embedding;

        if (parameters[eigen_method].is(Dense))
        {
            DenseMatrix distance_matrix_sym = distance_matrix * distance_matrix.transpose();
            landmarks_embedding = eigendecomposition(parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues,
                                                     distance_matrix_sym, parameters[target_dimension]);
        }
        else
        {
            landmarks_embedding = eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SquaredLargestEigenvalues,
                                                     distance_matrix, parameters[target_dimension]);
        }

        DenseMatrix embedding = distance_matrix.transpose() * landmarks_embedding.first;

        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }

    TapkeeOutput embedNeighborhoodPreservingEmbedding()
    {
        Neighbors neighbors = findNeighborsWith(kernel_distance);
        SparseWeightMatrix weight_matrix =
            linear_weight_matrix(begin, end, neighbors, kernel, parameters[nullspace_shift], parameters[klle_shift]);
        DenseSymmetricMatrixPair eig_matrices =
            construct_neighborhood_preserving_eigenproblem(weight_matrix, begin, end, features, current_dimension);
        EigendecompositionResult projection_result =
            generalized_eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues,
                                           eig_matrices.first, eig_matrices.second, parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }

    TapkeeOutput embedHessianLocallyLinearEmbedding()
    {
        Neighbors neighbors = findNeighborsWith(kernel_distance);
        SparseWeightMatrix weight_matrix = hessian_weight_matrix(begin, end, neighbors, kernel, parameters[target_dimension]);
        return TapkeeOutput(eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues,
                                               weight_matrix, parameters[target_dimension])
                                .first,
                            unimplementedProjectingFunction());
    }

    TapkeeOutput embedLaplacianEigenmaps()
    {
        Neighbors neighbors = findNeighborsWith(plain_distance);
        Laplacian laplacian = compute_laplacian(begin, end, neighbors, distance, parameters[gaussian_kernel_width]);
        return TapkeeOutput(generalized_eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues,
                                                           laplacian.first, laplacian.second, parameters[target_dimension])
                                .first,
                            unimplementedProjectingFunction());
    }

    TapkeeOutput embedLocalityPreservingProjections()
    {
        Neighbors neighbors = findNeighborsWith(plain_distance);
        Laplacian laplacian = compute_laplacian(begin, end, neighbors, distance, parameters[gaussian_kernel_width]);
        DenseSymmetricMatrixPair eigenproblem_matrices = construct_locality_preserving_eigenproblem(
            laplacian.first, laplacian.second, begin, end, features, current_dimension);
        EigendecompositionResult projection_result = generalized_eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues, eigenproblem_matrices.first,
            eigenproblem_matrices.second, parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }

    TapkeeOutput embedPCA()
    {
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        DenseSymmetricMatrix centered_covariance_matrix =
            compute_covariance_matrix(begin, end, mean_vector, features, current_dimension);
        EigendecompositionResult projection_result = eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues, centered_covariance_matrix, parameters[target_dimension]);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }

    TapkeeOutput embedRandomProjection()
    {
        DenseMatrix projection_matrix = gaussian_projection_matrix(current_dimension, parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);

        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_matrix, mean_vector));
        return TapkeeOutput(project(projection_matrix, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }

    TapkeeOutput embedKernelPCA()
    {
        DenseSymmetricMatrix centered_kernel_matrix = compute_centered_kernel_matrix(begin, end, kernel);
        EigendecompositionResult embedding = eigendecomposition(
            parameters[eigen_method], parameters[computation_strategy], LargestEigenvalues, centered_kernel_matrix, parameters[target_dimension]);
        for (IndexType i = 0; i < static_cast<IndexType>(parameters[target_dimension]); i++)
            embedding.first.col(i).array() *= sqrt(embedding.second(i));
        return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
    }

    TapkeeOutput embedLinearLocalTangentSpaceAlignment()
    {
        Neighbors neighbors = findNeighborsWith(kernel_distance);
        SparseWeightMatrix weight_matrix =
            tangent_weight_matrix(begin, end, neighbors, kernel, parameters[target_dimension], parameters[nullspace_shift]);
        DenseSymmetricMatrixPair eig_matrices =
            construct_lltsa_eigenproblem(weight_matrix, begin, end, features, current_dimension);
        EigendecompositionResult projection_result =
            generalized_eigendecomposition(parameters[eigen_method], parameters[computation_strategy], SmallestEigenvalues,
                                           eig_matrices.first, eig_matrices.second, parameters[target_dimension]);
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        tapkee::ProjectingFunction projecting_function(
            new tapkee::MatrixProjectionImplementation(projection_result.first, mean_vector));
        return TapkeeOutput(project(projection_result.first, mean_vector, begin, end, features, current_dimension),
                            projecting_function);
    }

    TapkeeOutput embedStochasticProximityEmbedding()
    {
        Neighbors neighbors;
        if (parameters[spe_global_strategy].is(false))
        {
            neighbors = findNeighborsWith(plain_distance);
        }

        return TapkeeOutput(spe_embedding(begin, end, distance, neighbors, parameters[target_dimension], parameters[spe_global_strategy],
                                          parameters[spe_tolerance], parameters[spe_num_updates], parameters[max_iteration]),
                            unimplementedProjectingFunction());
    }

    TapkeeOutput embedPassThru()
    {
        DenseMatrix feature_matrix = dense_matrix_from_features(features, current_dimension, begin, end);
        return TapkeeOutput(feature_matrix.transpose(), unimplementedProjectingFunction());
    }

    TapkeeOutput embedFactorAnalysis()
    {
        DenseVector mean_vector = compute_mean(begin, end, features, current_dimension);
        return TapkeeOutput(project(begin, end, features, current_dimension, parameters[max_iteration], parameters[fa_epsilon],
                                    parameters[target_dimension], mean_vector),
                            unimplementedProjectingFunction());
    }

    TapkeeOutput embedtDistributedStochasticNeighborEmbedding()
    {
        parameters[sne_perplexity].checked().satisfies(InClosedRange<ScalarType>(0.0, (n_vectors - 1) / 3.0)).orThrow();

        DenseMatrix data = dense_matrix_from_features(features, current_dimension, begin, end);

        DenseMatrix embedding(static_cast<IndexType>(parameters[target_dimension]), n_vectors);
        tsne::TSNE tsne;
        tsne.run(data, data.cols(), data.rows(), embedding.data(), parameters[target_dimension], parameters[sne_perplexity], parameters[sne_theta]);

        return TapkeeOutput(embedding.transpose(), unimplementedProjectingFunction());
    }

    TapkeeOutput embedManifoldSculpting()
    {
        parameters[squishing_rate].checked().satisfies(InRange<ScalarType>(0.0, 1.0)).orThrow();

        DenseMatrix embedding = dense_matrix_from_features(features, current_dimension, begin, end);

        Neighbors neighbors = findNeighborsWith(plain_distance);

        manifold_sculpting_embed(begin, end, embedding, parameters[target_dimension], neighbors, distance, parameters[max_iteration],
                                 parameters[squishing_rate]);

        return TapkeeOutput(embedding, unimplementedProjectingFunction());
    }
};

template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>
ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback> initialize(
    RandomAccessIterator begin, RandomAccessIterator end, KernelCallback kernel, DistanceCallback distance,
    FeaturesCallback features, stichwort::ParametersSet& pmap, const Context& ctx)
{
    return ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>(
        begin, end, kernel, distance, features, pmap, ctx);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
