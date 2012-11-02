/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef EDRT_MAIN_H_
#define EDRT_MAIN_H_

#define HAVE_LAPACK
#define HAVE_ARPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack.h>
#undef HAVE_LAPACK
#undef HAVE_ARPACK
#include <shogun/lib/Time.h>
#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <algorithm>
#include <iostream>

#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SuperLUSupport>
#include "defines.hpp"
#include "methods/local_weights.hpp"
#include "neighbors/neighbors.hpp"
#include "methods/eigen_embedding.hpp"

template <class RandomAccessIterator, class PairwiseCallback>
DenseMatrix embed(const RandomAccessIterator& begin, const RandomAccessIterator& end,
                  const PairwiseCallback& callback, ParametersMap options)
{
	Neighbors neighbors;
	WeightMatrix weight_matrix;
	EmbeddingMatrix embedding_matrix;

	EDRT_METHOD method = options[REDUCTION_METHOD].cast<EDRT_METHOD>();
	EDRT_EIGEN_EMBEDDING_METHOD eigen_method = options[EIGEN_EMBEDDING_METHOD].cast<EDRT_EIGEN_EMBEDDING_METHOD>();
	EDRT_NEIGHBORS_METHOD neighbors_method = options[NEIGHBORS_METHOD].cast<EDRT_NEIGHBORS_METHOD>();
	unsigned int target_dimension = options[TARGET_DIMENSIONALITY].cast<unsigned int>();
	unsigned int k = options[NUMBER_OF_NEIGHBORS].cast<unsigned int>();

	switch (method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			{
				timed_context context("Embedding with KLLE");
				neighbors = find_neighbors(neighbors_method,begin,end,callback,k);
				weight_matrix = klle_weight_matrix(begin,end,neighbors,callback);
				embedding_matrix = eigen_embedding(eigen_method,weight_matrix,target_dimension);
			}
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			{
				timed_context context("Embedding with KLTSA");
				neighbors = find_neighbors(neighbors_method,begin,end,callback,k);
				weight_matrix = kltsa_weight_matrix(begin,end,neighbors,callback,target_dimension);
				embedding_matrix = eigen_embedding(eigen_method,weight_matrix,target_dimension);
			}
			break;
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			break;
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			break;
		case LAPLACIAN_EIGENMAPS:
			break;
		case LOCALITY_PRESERVING_PROJECTIONS:
			break;
		case DIFFUSION_MAPS:
			break;
		case ISOMAP:
			break;
		case MULTIDIMENSIONAL_SCALING:
			break;
		default:
			break;
	}
	return embedding_matrix;
}

#endif /* libedrt_h_ */
