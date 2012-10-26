#include "libedrt.hpp"
#include "libedrt_methods.hpp"
#include "libedrt_neighbors.hpp"
#include "libedrt_embedding.hpp"

#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include <eigen3/Eigen/SparseCore>

template <class PairwiseCallback, class ResultInsertIterator>
int embed(
		const edrt_options_t& options,
		const int target_dimension,
		const int N,
		const int dimension,
		const int k)
{
	typedef std::vector<int> LocalNeighbors;
	typedef std::vector<LocalNeighbors> Neighbors;
	typedef Eigen::SparseMatrix<double> WeightMatrix;

	Neighbors neighbors;
	WeightMatrix weight_matrix;

	switch (options.method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
/*
			neighborhood_matrix = neighbors_matrix(N, k, NULL, kernel, user_data);
			REQUIRE(neighborhood_matrix);

			weight_matrix = klle_weight_matrix(neighborhood_matrix, N, k, k, 
			                                   options.num_threads, 
			                                   options.klle_reconstruction_shift,
			                                   kernel, user_data);
			REQUIRE(weight_matrix);

			*output = eigendecomposition_embedding(weight_matrix, N, 
			                                       target_dimension, 
			                                       options.use_arpack,
			                                       options.nullspace_shift);
*/
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
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
	}
	return 0;
}



