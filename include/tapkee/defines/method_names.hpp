/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

#include <tapkee/defines/methods.hpp>

#include <map>
#include <string>

namespace tapkee
{

inline const std::map<std::string, DimensionReductionMethod>& dimension_reduction_methods()
{
    static const std::map<std::string, DimensionReductionMethod> methods = {
        {"local_tangent_space_alignment", KernelLocalTangentSpaceAlignment},
        {"ltsa", KernelLocalTangentSpaceAlignment},
        {"locally_linear_embedding", KernelLocallyLinearEmbedding},
        {"lle", KernelLocallyLinearEmbedding},
        {"hessian_locally_linear_embedding", HessianLocallyLinearEmbedding},
        {"hlle", HessianLocallyLinearEmbedding},
        {"multidimensional_scaling", MultidimensionalScaling},
        {"mds", MultidimensionalScaling},
        {"landmark_multidimensional_scaling", LandmarkMultidimensionalScaling},
        {"l-mds", LandmarkMultidimensionalScaling},
        {"isomap", Isomap},
        {"landmark_isomap", LandmarkIsomap},
        {"l-isomap", LandmarkIsomap},
        {"diffusion_map", DiffusionMap},
        {"dm", DiffusionMap},
        {"kernel_pca", KernelPrincipalComponentAnalysis},
        {"kpca", KernelPrincipalComponentAnalysis},
        {"pca", PrincipalComponentAnalysis},
        {"random_projection", RandomProjection},
        {"ra", RandomProjection},
        {"laplacian_eigenmaps", LaplacianEigenmaps},
        {"la", LaplacianEigenmaps},
        {"locality_preserving_projections", LocalityPreservingProjections},
        {"lpp", LocalityPreservingProjections},
        {"neighborhood_preserving_embedding", NeighborhoodPreservingEmbedding},
        {"npe", NeighborhoodPreservingEmbedding},
        {"linear_local_tangent_space_alignment", LinearLocalTangentSpaceAlignment},
        {"lltsa", LinearLocalTangentSpaceAlignment},
        {"stochastic_proximity_embedding", StochasticProximityEmbedding},
        {"spe", StochasticProximityEmbedding},
        {"passthru", PassThru},
        {"factor_analysis", FactorAnalysis},
        {"fa", FactorAnalysis},
        {"t-stochastic_proximity_embedding", tDistributedStochasticNeighborEmbedding},
        {"t-sne", tDistributedStochasticNeighborEmbedding},
        {"manifold_sculpting", ManifoldSculpting},
    };
    return methods;
}

} // namespace tapkee
