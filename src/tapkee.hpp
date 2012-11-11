/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_MAIN_H_
#define TAPKEE_MAIN_H_

#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <algorithm>
#include <iostream>

#include "defines.hpp"
#include "methods/local_weights.hpp"
#include "methods/eigen_embedding.hpp"
#include "methods/generalized_eigen_embedding.hpp"
#include "methods/multidimensional_scaling.hpp"
#include "methods/diffusion_maps.hpp"
#include "methods/laplacian_eigenmaps.hpp"
#include "methods/isomap.hpp"
#include "methods/pca.hpp"
#include "neighbors/neighbors.hpp"

/** Main entry-point of the library. Constructs dense embedding with specified dimension
 * using provided data and callbacks.
 *
 * Has four template parameters:
 * 
 * RandomAccessIterator basic random access iterator with no specific capabilities.
 * 
 * KernelCallback that defines DefaultScalarType operator()(RandomAccessIterator, RandomAccessIterator) operation 
 * between two iterators. The operation should return value of Mercer kernel function 
 * between vectors/objects iterators pointing to. KernelCallback should be marked as a kernel function using
 * TAPKEE_CALLBACK_IS_KERNEL macro (fails during compilation in other case).
 * 
 * DistanceCallback that defines DefaultScalarType operator()(RandomAccessIterator, RandomAccessIterator) operation
 * between two iterators. DistanceCallback should be marked as a distance function using 
 * TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
 * 
 * FeatureVectorCallback TODO
 *
 * Parameters required by the chosen algorithm are obtained from the parameter map. It fails during runtime if
 * some of required parameters are not specified or have improper values.
 *
 * @param begin begin iterator of data
 * @param end end iterator of data
 * @param kernel_callback the kernel callback described before
 * @param distance_callback the distance callback described before
 * @param feature_vector_callback TODO
 * @param options parameter map
 */
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
DenseMatrix embed(RandomAccessIterator begin, RandomAccessIterator end,
                  KernelCallback kernel_callback, DistanceCallback distance_callback,
                  FeatureVectorCallback feature_vector_callback, ParametersMap options)
{
	Eigen::initParallel();
	EmbeddingResult embedding_result;

	// load common parameters from the parameters map
	TAPKEE_METHOD method = 
		options[REDUCTION_METHOD].cast<TAPKEE_METHOD>();
	TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method = 
		options[EIGEN_EMBEDDING_METHOD].cast<TAPKEE_EIGEN_EMBEDDING_METHOD>();
	unsigned int target_dimension = 
		options[TARGET_DIMENSION].cast<unsigned int>();

	switch (method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			{
				timed_context context("Embedding with KLLE");
				unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
				TAPKEE_NEIGHBORS_METHOD neighbors_method = 
					options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
				// find neighbors of each vector
				Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
				// construct sparse weight matrix
				SparseWeightMatrix weight_matrix = klle_weight_matrix(begin,end,neighbors,kernel_callback);
				// construct embedding with eigendecomposition of the
				// sparse weight matrix
				embedding_result = 
					eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,weight_matrix,target_dimension,1);
			}
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			{
				timed_context context("Embedding with KLTSA");
				unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
				TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
				// find neighbors of each vector
				Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
				// construct sparse weight matrix
				SparseWeightMatrix weight_matrix = kltsa_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension);
				// construct embedding with eigendecomposition of the
				// sparse weight matrix
				embedding_result = 
					eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,weight_matrix,target_dimension,1);
			}
			break;
		case DIFFUSION_MAP:
			{
				timed_context context("Embedding with diffusion map");
				unsigned int timesteps = options[DIFFUSION_MAP_TIMESTEPS].cast<unsigned int>();
				DefaultScalarType width = options[GAUSSIAN_KERNEL_WIDTH].cast<DefaultScalarType>();
				// compute diffusion matrix
				DenseSymmetricMatrix diffusion_matrix = compute_diffusion_matrix(begin,end,distance_callback,timesteps,width);
				// compute embedding with eigendecomposition
				embedding_result = 
					eigen_embedding<DenseSymmetricMatrix, DenseImplicitSquareMatrixOperation>(eigen_method,
							diffusion_matrix,target_dimension,0);
			}
			break;
		case MULTIDIMENSIONAL_SCALING:
			{
				timed_context context("Embeding with MDS");
				// compute distance matrix (matrix of pairwise distances) of data
				DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance_callback);
				// process the distance matrix (center it and *(-0.5))
				mds_process_matrix(distance_matrix);
				// construct embedding with eigendecomposition of the
				// dense weight matrix
				embedding_result = 
					eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,distance_matrix,target_dimension,0);
			}
			break;
		case LANDMARK_MULTIDIMENSIONAL_SCALING:
			{
				timed_context context("Embedding with landmark MDS");
			}
			break;
		case ISOMAP:
			{
				timed_context context("Embedding with Isomap");
				unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
				TAPKEE_NEIGHBORS_METHOD neighbors_method = 
					options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
				// find neighbors of each vector
				Neighbors neighbors = find_neighbors(neighbors_method,begin,end,distance_callback,k);
				// compute distance matrix (matrix of pairwise distances) of data
				DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance_callback);
				// relax distances with Dijkstra shortest path algorithm
				DenseSymmetricMatrix relaxed_distance_matrix = isomap_relax_distances(distance_matrix,neighbors);
				// construct embedding with eigendecomposition of the
				// dense weight matrix
				embedding_result = 
					eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,relaxed_distance_matrix,target_dimension,0);
			}
			break;
		case LANDMARK_ISOMAP:
			{
				timed_context context("Embedding with landmark Isomap");
			}
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			{
				timed_context context("Embedding with NPE");
			}
			break;
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			{
				timed_context context("Embedding with NPE");
			}
			break;
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			{
				timed_context context("Embedding with HLLE");
				unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
				TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
				// find neighbors of each vector
				Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
				// construct sparse weight matrix
				SparseWeightMatrix weight_matrix = hlle_weight_matrix(begin,end,neighbors,kernel_callback,target_dimension);
				// construct embedding with eigendecomposition of the
				// sparse weight matrix
				embedding_result = 
					eigen_embedding<SparseWeightMatrix,InverseSparseMatrixOperation>(eigen_method,weight_matrix,target_dimension,1);
			}
			break;
		case LAPLACIAN_EIGENMAPS:
			{
				timed_context context("Embedding with Laplacian Eigenmaps");
				unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();
				TAPKEE_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<TAPKEE_NEIGHBORS_METHOD>();
				DefaultScalarType width = options[GAUSSIAN_KERNEL_WIDTH].cast<DefaultScalarType>();
				// find neighbors of each vector
				Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_callback,k);
				// construct sparse weight matrix
				pair<SparseWeightMatrix,DenseDiagonalMatrix> laplacian = 
					compute_laplacian(begin,end,neighbors,distance_callback,width);
				// construct embedding
				embedding_result = 
					generalized_eigen_embedding<SparseWeightMatrix,DenseDiagonalMatrix,InverseSparseMatrixOperation>(
							eigen_method,laplacian.first,laplacian.second,target_dimension,1);
			}
			break;
		case LOCALITY_PRESERVING_PROJECTIONS:
			{
				timed_context context("Embedding with LPP");
			}
			break;
		case PCA:
			{
				timed_context context("Embedding with PCA");

				unsigned int dimension = options[CURRENT_DIMENSION].cast<unsigned int>();
				// compute centered covariance matrix
				DenseSymmetricMatrix centered_covariance_matrix = compute_centered_covariance_matrix(begin,end,feature_vector_callback,dimension);
				
				ProjectionResult projection_result = 
					eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_covariance_matrix,target_dimension,0);
				// TODO to be improved with out-of-sample projection
				embedding_result = project(projection_result,begin,end,feature_vector_callback,dimension);
			}
			break;
		case KERNEL_PCA:
			{
				timed_context context("Embedding with kPCA");

				// compute centered kernel matrix 
				DenseSymmetricMatrix centered_kernel_matrix = compute_centered_kernel_matrix(begin,end,kernel_callback);
				// construct embedding
				embedding_result = 
					eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_kernel_matrix,target_dimension,0);
			}
			break;
		default:
			break;
	}
	return embedding_result.first;
}

#endif
