/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

#include "defines.hpp"
#include "methods/locally_linear.hpp"
#include "methods/eigen_embedding.hpp"
#include "methods/generalized_eigen_embedding.hpp"
#include "methods/multidimensional_scaling.hpp"
#include "methods/diffusion_maps.hpp"
#include "methods/laplacian_eigenmaps.hpp"
#include "methods/isomap.hpp"
#include "methods/pca.hpp"
#include "neighbors/neighbors.hpp"

template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback, int>
struct embedding_impl
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback distance_callback,
                          FeatureVectorCallback feature_vector_callback, ParametersMap options);
};

#define CONCRETE_IMPLEMENTATION(METHOD) \
	template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback> \
	struct embedding_impl<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,METHOD>

CONCRETE_IMPLEMENTATION(KERNEL_LOCALLY_LINEAR_EMBEDDING)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = 
			options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		
		timed_context context("Embedding with KLLE");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
		// construct sparse weight matrix
		SparseWeightMatrix weight_matrix = klle_weight_matrix(begin,end,neighbors,kernel_callback);
		// construct embedding with eigendecomposition of the
		// sparse weight matrix
		return eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,weight_matrix,target_dimension,1);
	}
};

CONCRETE_IMPLEMENTATION(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = 
			options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		
		timed_context context("Embedding with KLTSA");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
		// construct sparse weight matrix
		SparseWeightMatrix weight_matrix = kltsa_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension);
		// construct embedding with eigendecomposition of the
		// sparse weight matrix
		return eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,weight_matrix,target_dimension,1);
	}
};

CONCRETE_IMPLEMENTATION(DIFFUSION_MAP)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback distance_callback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int timesteps = options[DIFFUSION_MAP_TIMESTEPS].cast<unsigned int>();
		DefaultScalarType width = options[GAUSSIAN_KERNEL_WIDTH].cast<DefaultScalarType>();
		
		timed_context context("Embedding with diffusion map");
		// compute diffusion matrix
		DenseSymmetricMatrix diffusion_matrix = compute_diffusion_matrix(begin,end,distance_callback,timesteps,width);
		// compute embedding with eigendecomposition
		return eigen_embedding<DenseSymmetricMatrix, DenseImplicitSquareMatrixOperation>(eigen_method,
			diffusion_matrix,target_dimension,0);
	}
};

CONCRETE_IMPLEMENTATION(MULTIDIMENSIONAL_SCALING)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback distance_callback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();

		timed_context context("Embeding with MDS");
		// compute distance matrix (matrix of pairwise distances) of data
		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance_callback);
		// process the distance matrix (center it and *(-0.5))
		mds_process_matrix(distance_matrix);
		// construct embedding with eigendecomposition of the
		// dense weight matrix
		return eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,distance_matrix,target_dimension,0);
	}
};

CONCRETE_IMPLEMENTATION(ISOMAP)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback distance_callback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = 
			options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		
		timed_context context("Embedding with Isomap");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,distance_callback,k);
		// compute distance matrix (matrix of pairwise distances) of data
		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance_callback);
		// relax distances with Dijkstra shortest path algorithm
		DenseSymmetricMatrix relaxed_distance_matrix = isomap_relax_distances(distance_matrix,neighbors);
		// construct embedding with eigendecomposition of the
		// dense weight matrix
		return eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,relaxed_distance_matrix,target_dimension,0);
	}
};

CONCRETE_IMPLEMENTATION(NEIGHBORHOOD_PRESERVING_EMBEDDING)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		unsigned int dimension = options[CURRENT_DIMENSION].cast<unsigned int>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		
		timed_context context("Embedding with NPE");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
		// construct sparse weight matrix
		SparseWeightMatrix weight_matrix = klle_weight_matrix(begin,end,neighbors,kernel_callback);
		// 
		pair<DenseSymmetricMatrix,DenseSymmetricMatrix> eigenproblem_matrices =
			construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
					feature_vector_callback,dimension);
		// construct embedding
		ProjectionResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseMatrixOperation>(
					eigen_method,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension,1);
		// TODO to be improved with out-of-sample projection
		return project(projection_result,begin,end,feature_vector_callback,dimension);
	}
};

CONCRETE_IMPLEMENTATION(HESSIAN_LOCALLY_LINEAR_EMBEDDING)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		
		timed_context context("Embedding with HLLE");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
		// construct sparse weight matrix
		SparseWeightMatrix weight_matrix = hlle_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension);
		// construct embedding with eigendecomposition of the
		// sparse weight matrix
		return eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,weight_matrix,target_dimension,1);
	}
};

CONCRETE_IMPLEMENTATION(LAPLACIAN_EIGENMAPS)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback distance_callback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		DefaultScalarType width = options[GAUSSIAN_KERNEL_WIDTH].cast<DefaultScalarType>();
		
		timed_context context("Embedding with Laplacian Eigenmaps");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,distance_callback,k);
		// construct sparse weight matrix
		pair<SparseWeightMatrix,DenseDiagonalMatrix> laplacian = 
			compute_laplacian(begin,end,neighbors,distance_callback,width);
		// construct embedding
		return generalized_eigen_embedding<SparseWeightMatrix,DenseSymmetricMatrix,InverseSparseMatrixOperation>(
					eigen_method,laplacian.first,laplacian.second,target_dimension,1);
	}
};

CONCRETE_IMPLEMENTATION(LOCALITY_PRESERVING_PROJECTIONS)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback distance_callback,
                          FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
		unsigned int dimension = options[CURRENT_DIMENSION].cast<unsigned int>();
		TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
		DefaultScalarType width = options[GAUSSIAN_KERNEL_WIDTH].cast<DefaultScalarType>();
		
		timed_context context("Embedding with LPP");
		// find neighbors of each vector
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,distance_callback,k);
		// construct sparse weight matrix
		pair<SparseWeightMatrix,DenseDiagonalMatrix> laplacian = 
			compute_laplacian(begin,end,neighbors,distance_callback,width);
		pair<DenseSymmetricMatrix,DenseSymmetricMatrix> eigenproblem_matrices =
			construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
					feature_vector_callback,dimension);
		// construct embedding
		ProjectionResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseMatrixOperation>(
					eigen_method,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension,1);
		// TODO to be improved with out-of-sample projection
		return project(projection_result,begin,end,feature_vector_callback,dimension);
	}
};

CONCRETE_IMPLEMENTATION(PCA)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback, DistanceCallback,
                          FeatureVectorCallback feature_vector_callback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
		unsigned int dimension = options[CURRENT_DIMENSION].cast<unsigned int>();
		
		timed_context context("Embedding with PCA");
		// compute centered covariance matrix
		DenseSymmetricMatrix centered_covariance_matrix = compute_covariance_matrix(begin,end,feature_vector_callback,dimension);
		
		ProjectionResult projection_result = 
			eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_covariance_matrix,target_dimension,0);
		// TODO to be improved with out-of-sample projection
		return project(projection_result,begin,end,feature_vector_callback,dimension);
	}
};

CONCRETE_IMPLEMENTATION(KERNEL_PCA)
{
	EmbeddingResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                          KernelCallback kernel_callback, DistanceCallback,
                          FeatureVectorCallback, ParametersMap options)
	{
		unsigned int target_dimension = 
			options[TARGET_DIMENSION].cast<unsigned int>();
		TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
			options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();

		timed_context context("Embedding with kPCA");
		// compute centered kernel matrix 
		DenseSymmetricMatrix centered_kernel_matrix = compute_centered_kernel_matrix(begin,end,kernel_callback);
		// construct embedding
		return eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_kernel_matrix,target_dimension,0);
	}
};

#undef CONCRETE_IMPLEMENTATION
#endif
