#include "libedrt.hpp"
#include "libedrt_methods.hpp"
#include "libedrt_neighbors.hpp"
#include "libedrt_embedding.hpp"

#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>

#include "../neighbor/CoverTree.hpp"
#include <eigen3/Eigen/SparseCore>

typedef std::vector<int> LocalNeighbors;
typedef std::vector<LocalNeighbors> Neighbors;
typedef Eigen::SparseMatrix<double> WeightMatrix;
typedef Eigen::MatrixXd EmbeddingMatrix;

template <class ForwardIterator, class PairwiseCallback>
Neighbors find_neighbors(ForwardIterator begin, ForwardIterator end, PairwiseCallback callback, unsigned int k)
{
}

template <class ForwardIterator, class PairwiseCallback>
WeightMatrix klle_weight_matrix(ForwardIterator begin, ForwardIterator end, PairwiseCallback callback)
{

}

EmbeddingMatrix eigen_embedding(WeightMatrix wm, unsigned int target_dimension)
{

}

template <class ForwardIterator, class PairwiseCallback, class ResultInsertIterator>
int embed(
		ForwardIterator begin,
		ForwardIterator end,
		const edrt_options_t& options,
		const int target_dimension,
		const int N,
		const int dimension,
		const int k)
{
	Neighbors neighbors;
	WeightMatrix weight_matrix;
	EmbeddingMatrix embedding_matrix;

	switch (options.method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			PairwiseCallback kernel_callback = PairwiseCallback();
			neighbors = find_neighbors(begin,end,kernel_callback,k);
			weight_matrix = klle_weight_matrix(begin,end,neighbors,kernel_callback);
			embedding_matrix = eigen_embedding(weight_matrix,target_dimension);
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



