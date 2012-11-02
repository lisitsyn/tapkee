#ifndef libedrt_h_
#define libedrt_h_

#define HAVE_LAPACK
#define HAVE_ARPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack.h>
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
DenseMatrix embed(
		RandomAccessIterator begin,
		RandomAccessIterator end,
		PairwiseCallback callback,
		ParametersMap options)
{
	Neighbors neighbors;
	WeightMatrix weight_matrix;
	EmbeddingMatrix embedding_matrix;

	EDRT_METHOD method = options[REDUCTION_METHOD].cast<EDRT_METHOD>();
	int target_dimension = options[TARGET_DIMENSIONALITY].cast<int>();
	int k = options[NUMBER_OF_NEIGHBORS].cast<int>();

	switch (method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			{
				timed_context context("Embedding with KLLE");
				neighbors = find_neighbors(begin,end,callback,k);
				weight_matrix = klle_weight_matrix(begin,end,neighbors,callback);
				embedding_matrix = eigen_embedding<randomized_shift_inverse>()(weight_matrix,target_dimension);
			}
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			{
				timed_context context("Embedding with KLTSA");
				neighbors = find_neighbors(begin,end,callback,k);
				weight_matrix = kltsa_weight_matrix(begin,end,neighbors,callback,target_dimension);
				embedding_matrix = eigen_embedding<randomized_shift_inverse>()(weight_matrix,target_dimension);
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
