#include "libedrt.hpp"
#include "libedrt_methods.hpp"
#include "libedrt_neighbors.hpp"
#include "libedrt_embedding.hpp"

#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <algorithm>

#include "../neighbor/CoverTree.hpp"
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SuperLUSupport>

typedef std::vector<int> LocalNeighbors;
typedef std::vector<LocalNeighbors> Neighbors;
typedef Eigen::SparseMatrix<double> WeightMatrix;
typedef Eigen::MatrixXd EmbeddingMatrix;

template <class RandomAccessIterator, class PairwiseCallback>
Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback callback, unsigned int k)
{
	typedef std::pair<RandomAccessIterator, double> DistanceRecord;
	typedef std::vector<DistanceRecord> Distances;
	
	struct distances_comparator
	{
		bool operator()(const DistanceRecord& l, const DistanceRecord& r) 
		{
			return (l.second < r.second);
		}
	};

	Neighbors neighbors;
	neighbors.reserve(end-begin);
	Distances distances;
	distances.reserve(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		distances.clear();
		for (RandomAccessIterator around_iter=begin; around_iter!=end; ++around_iter)
			distances.push_back(make_pair(around_iter, callback(iter,around_iter)));
		
		std::partial_sort(distances.begin(),distances.begin()+k+1,distances.end(),distances_comparator());

		LocalNeighbors local_neighbors;
		local_neighbors.reserve(k);
		for (typename DistanceRecord::const_iterator neighbors_iter=distances.begin()+1; 
				neighbors_iter!=distances.begin()+k+1; ++neighbors_iter)
			local_neighbors.push_back(neighbors_iter->first - begin);
		neighbors.push_back(local_neighbors);
	}

	return neighbors;
}

template <class RandomAccessIterator, class PairwiseCallback>
WeightMatrix klle_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end, Neighbors neighbors, PairwiseCallback callback)
{
	typedef Eigen::VectorXd DenseVector;
	typedef Eigen::MatrixXd DenseMatrix;
	typedef Eigen::Triplet<double> SparseTriplet;
	typedef std::vector<SparseTriplet> SparseTriplets;
	
	int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(begin-end));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
#pragma omp parallel for private(iter) shared(iter_begin, iter_end)
	for (RandomAccessIterator iter=iter_begin; iter!=iter_end; ++iter)
	{
		double kernel_value = callback(iter,iter);
		LocalNeighbors& current_neighbors = neighbors[iter-begin];
		
		DenseVector dots(k);
		for (int i=0; i<k; ++i)
			dots[i] = callback(iter, begin+current_neighbors[i]);

		DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
		gram_matrix += kernel_value;
		for (int i=0; i<k; ++i)
		{
			gram_matrix.row(i) -= dots(i);
			gram_matrix.col(i) -= dots(i);
			for (int j=0; j<k; ++j)
				gram_matrix(i,j) -= callback(begin+i,begin+j);
		}
		gram_matrix.diagonal() += 1e-3*gram_matrix.trace();
		DenseVector rhs = DenseVector::Ones(k);
		DenseVector weights = gram_matrix.ldlt().solve(rhs);
		weights /= weights.sum();

		sparse_triplets.push_back(SparseTriplet(iter-begin,iter-begin,1.0));
		for (int i=0; i<k; ++i)
		{
			sparse_triplets.push_back(SparseTriplet(current_neighbors[i],iter-begin,
			                                        -weights[i]));
			sparse_triplets.push_back(SparseTriplet(iter-begin,current_neighbors[i],
			                                        -weights[i]));
			for (int j=0; j<k; ++j)
				sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[j],
				                                        +gram_matrix(i,j)));
		}
	}

	WeightMatrix weight_matrix(begin-end,begin-end);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());
	return weight_matrix;
}

EmbeddingMatrix eigen_embedding(WeightMatrix wm, unsigned int target_dimension)
{
	Eigen::MatrixXd O(wm.rows(), target_dimension);
	for (int i=0; i<O.rows(); ++i)
	{
		int j=0;
		for ( ; j+1 < O.cols(); j+= 2)
		{
			double v1 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double v2 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double len = sqrt(-2.f*log(v1));
			O(i,j) = len*cos(2.f*M_PI*v2);
			O(i,j+1) = len*sin(2.f*M_PI*v2);
		}
		for ( ; j < O.cols(); j++)
		{
			double v1 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double v2 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double len = sqrt(-2.f*log(v1));
			O(i,j) = len*cos(2.f*M_PI*v2);
		}
	}
	Eigen::SuperLU<WeightMatrix> solver;
	solver.compute(wm);

	Eigen::MatrixXd Y = solver.solve(O);
	for (int i=0; i<Y.cols(); i++)
	{
		for (int j=0; j<i; j++)
		{
			double r = Y.col(i).dot(Y.col(j));
			Y.col(i) -= r*Y.col(j);
		}
		double norm = Y.col(i).norm();
		if (norm < 1e-4)
		{
			for (int k = i; k<Y.cols(); k++)
				Y.col(k).setZero();
		}
		Y.col(i) *= (1.f / norm);
	}

	Eigen::MatrixXd B1 = solver.solve(Y);
	Eigen::MatrixXd B = Y.partialPivLu().solve(B1);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenOfB(B);
	return Y*eigenOfB.eigenvectors();
}

template <class RandomAccessIterator, class PairwiseCallback, class ResultInsertIterator>
int embed(
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



