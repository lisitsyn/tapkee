#ifndef libedrt_methods_h_
#define libedrt_methods_h_

#include "../defines.hpp"
#include <iostream>
#include <shogun/lib/SGMatrix.h>
#include "../utils/time.hpp"

using std::cout;
using std::endl;

template <class RandomAccessIterator, class PairwiseCallback>
WeightMatrix kltsa_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                Neighbors neighbors, PairwiseCallback callback, unsigned int target_dimension)
{
	timed_context context("KLTSA weight matrix computation");
	int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(end-begin));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector col_means(k), row_means(k);
	DenseVector rhs = DenseVector::Ones(k);
	WeightMatrix weight_matrix(end-begin,end-begin);
	for (RandomAccessIterator iter=iter_begin; iter!=iter_end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
	
		for (int i=0; i<k; ++i)
		{
			for (int j=i; j<k; ++j)
			{
				double kij = callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
				gram_matrix(i,j) = kij;
				gram_matrix(j,i) = kij;
			}
		}
		
		for (int i=0; i<k; ++i)
		{
			col_means[i] = gram_matrix.col(i).mean();
			row_means[i] = gram_matrix.row(i).mean();
		}
		double grand_mean = gram_matrix.mean();
		gram_matrix.array() += grand_mean;
		gram_matrix.rowwise() -= col_means.transpose();
		gram_matrix.colwise() -= row_means;
		
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sae_solver;
		sae_solver.compute(gram_matrix);

		DenseMatrix G = DenseMatrix::Zero(k,target_dimension+1);
		G.col(0).setConstant(1/sqrt(k));

		G.rightCols(target_dimension) = sae_solver.eigenvectors().rightCols(target_dimension);
		gram_matrix = G * G.transpose();
		
		sparse_triplets.push_back(SparseTriplet(iter-begin,iter-begin,1e-8));
		for (int i=0; i<k; ++i)
		{
			sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[i],1.0));
			for (int j=0; j<k; ++j)
				sparse_triplets.push_back(SparseTriplet(current_neighbors[i],current_neighbors[j],
				                                        -gram_matrix(i,j)));
		}
	}

	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}
template <class RandomAccessIterator, class PairwiseCallback>
WeightMatrix klle_weight_matrix(RandomAccessIterator begin, RandomAccessIterator end, 
                                Neighbors neighbors, PairwiseCallback callback)
{
	timed_context context("KLLE weight computation");
	int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(end-begin));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector dots(k);
	DenseVector rhs = DenseVector::Ones(k);
	DenseVector weights;
	for (RandomAccessIterator iter=iter_begin; iter!=iter_end; ++iter)
	{
		double kernel_value = callback(*iter,*iter);
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
		
		for (int i=0; i<k; ++i)
			dots[i] = callback(*iter, begin[current_neighbors[i]]);

		for (int i=0; i<k; ++i)
		{
			for (int j=0; j<k; ++j)
				gram_matrix(i,j) = kernel_value - dots(i) - dots(j) + callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
		}
		
		double trace = gram_matrix.trace();
		gram_matrix.diagonal().array() += 1e-3*trace;
		weights = gram_matrix.ldlt().solve(rhs);
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
				                                        +weights(i)*weights(j)));
		}
	}

	WeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}

#endif
