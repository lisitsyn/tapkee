/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef EDRT_LOCAL_METHODS_H_
#define EDRT_LOCAL_METHODS_H_

#include "../defines.hpp"
#include "../utils/time.hpp"

template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix kltsa_weight_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end, 
                                       const Neighbors& neighbors, const PairwiseCallback& callback, unsigned int target_dimension)
{
	timed_context context("KLTSA weight matrix computation");
	const int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(end-begin));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector col_means(k), row_means(k);
	DenseVector rhs = DenseVector::Ones(k);
	DenseMatrix G = DenseMatrix::Zero(k,target_dimension+1);
	for (RandomAccessIterator iter=iter_begin; iter!=iter_end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
	
		for (int i=0; i<k; ++i)
		{
			for (int j=i; j<k; ++j)
			{
				DefaultScalarType kij = callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
				gram_matrix(i,j) = kij;
				gram_matrix(j,i) = kij;
			}
		}
		
		for (int i=0; i<k; ++i)
		{
			col_means[i] = gram_matrix.col(i).mean();
			row_means[i] = gram_matrix.row(i).mean();
		}
		DefaultScalarType grand_mean = gram_matrix.mean();
		gram_matrix.array() += grand_mean;
		gram_matrix.rowwise() -= col_means.transpose();
		gram_matrix.colwise() -= row_means;
		
		Eigen::SelfAdjointEigenSolver<DenseMatrix> sae_solver;
		sae_solver.compute(gram_matrix);

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

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}

template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix klle_weight_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end, 
                                      const Neighbors& neighbors, const PairwiseCallback& callback)
{
	timed_context context("KLLE weight computation");
	const int k = neighbors[0].size();

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
		DefaultScalarType kernel_value = callback(*iter,*iter);
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
		
		for (int i=0; i<k; ++i)
			dots[i] = callback(*iter, begin[current_neighbors[i]]);

		for (int i=0; i<k; ++i)
		{
			for (int j=0; j<k; ++j)
				gram_matrix(i,j) = kernel_value - dots(i) - dots(j) + callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
		}
		
		DefaultScalarType trace = gram_matrix.trace();
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

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}

	template <class RandomAccessIterator, class PairwiseCallback>
SparseWeightMatrix hlle_weight_matrix(const RandomAccessIterator& begin, const RandomAccessIterator& end, 
                                      const Neighbors& neighbors, const PairwiseCallback& callback, unsigned int target_dimension)
{
	timed_context context("KLTSA weight matrix computation");
	const int k = neighbors[0].size();

	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*k*(end-begin));

	RandomAccessIterator iter;
	RandomAccessIterator iter_begin = begin, iter_end = end;
	DenseMatrix gram_matrix = DenseMatrix::Zero(k,k);
	DenseVector col_means(k), row_means(k);
	DenseVector rhs = DenseVector::Ones(k);

//	unsigned int dp = target_dimension*(target_dimension+1)/2;

	for (RandomAccessIterator iter=iter_begin; iter!=iter_end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
	
		for (int i=0; i<k; ++i)
		{
			for (int j=i; j<k; ++j)
			{
				DefaultScalarType kij = callback(begin[current_neighbors[i]],begin[current_neighbors[j]]);
				gram_matrix(i,j) = kij;
				gram_matrix(j,i) = kij;
			}
		}
		
		for (int i=0; i<k; ++i)
		{
			col_means[i] = gram_matrix.col(i).mean();
			row_means[i] = gram_matrix.row(i).mean();
		}
		DefaultScalarType grand_mean = gram_matrix.mean();
		gram_matrix.array() += grand_mean;
		gram_matrix.rowwise() -= col_means.transpose();
		gram_matrix.colwise() -= row_means;
		
		Eigen::SelfAdjointEigenSolver<DenseMatrix> sae_solver;
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

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return weight_matrix;
}

#endif
