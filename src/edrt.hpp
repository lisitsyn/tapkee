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

using std::cout;
using std::endl;


enum edrt_method_t
{
	KERNEL_LOCALLY_LINEAR_EMBEDDING,
	NEIGHBORHOOD_PRESERVING_EMBEDDING,
	KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT,
	LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT,
	HESSIAN_LOCALLY_LINEAR_EMBEDDING,
	LAPLACIAN_EIGENMAPS,
	LOCALITY_PRESERVING_PROJECTIONS,
	DIFFUSION_MAPS,
	ISOMAP,
	MULTIDIMENSIONAL_SCALING,
	STOCHASTIC_PROXIMITY_EMBEDDING,
	MAXIMUM_VARIANCE_UNFOLDING
};

struct edrt_options_t
{
	edrt_options_t()
	{
		method = KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
		num_threads = 1;
		use_arpack = true;
		use_superlu = true;
		mds_use_landmarks = false;
		klle_reconstruction_shift = 1e-3;
		diffusion_maps_t = 1;
		nullspace_shift = 1e-9;
	}
	// EDRT method
	edrt_method_t method;
	// number of threads
	int num_threads;
	// true if ARPACK should be used
	bool use_arpack;
	// true if SuperLU should be used
	bool use_superlu;
	// mds use landmarks
	bool mds_use_landmarks;
	// kernel LLE reconstruction shift
	double klle_reconstruction_shift;
	// diffusion maps t
	int diffusion_maps_t;
	// nullspace regularization shift
	double nullspace_shift;
};

template <class RandomAccessIterator, class PairwiseCallback>
Eigen::MatrixXd embed(
		RandomAccessIterator begin,
		RandomAccessIterator end,
		PairwiseCallback callback,
		const edrt_options_t& options,
		const int target_dimension,
		const int dimension,
		const int k)
{
	Neighbors neighbors;
	WeightMatrix weight_matrix;
	EmbeddingMatrix embedding_matrix;

	switch (options.method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			neighbors = find_neighbors(begin,end,callback,k);
			weight_matrix = klle_weight_matrix(begin,end,neighbors,callback);
			embedding_matrix = eigen_embedding<randomized_shift_inverse>()(weight_matrix,target_dimension);
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			neighbors = find_neighbors(begin,end,callback,k);
			weight_matrix = kltsa_weight_matrix(begin,end,neighbors,callback,target_dimension);
			embedding_matrix = eigen_embedding<arpack_dsxupd>()(weight_matrix,target_dimension);
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
