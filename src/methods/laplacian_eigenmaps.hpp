/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_LAPLACIAN_EIGENMAPS_H_
#define TAPKEE_LAPLACIAN_EIGENMAPS_H_
	
#include "../defines.hpp"
#include "../utils/time.hpp"
#include <utility>
using std::pair;
using std::make_pair;

template<class RandomAccessIterator, class DistanceCallback>
pair<SparseWeightMatrix,DenseDiagonalMatrix> compute_laplacian(RandomAccessIterator begin, 
		RandomAccessIterator end,const Neighbors& neighbors, 
		DistanceCallback callback, DefaultScalarType width)
{
	SparseTriplets sparse_triplets;

	timed_context context("Laplacian computation");
	const unsigned int k = neighbors[0].size();
	sparse_triplets.reserve(4*k*(end-begin));

	DenseVector D = DenseVector::Zero(end-begin);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];

		for (unsigned int i=0; i<k; ++i)
		{
			DefaultScalarType distance = callback(*iter,begin[current_neighbors[i]]);
			DefaultScalarType heat = exp(-distance*distance/width);
			D(iter-begin) += heat;
			//sparse_triplets.push_back(SparseTriplet(begin[current_neighbors[i]],(iter-begin),-heat));
			sparse_triplets.push_back(SparseTriplet((iter-begin),begin[current_neighbors[i]],-heat));
		}
	}
	for (unsigned int i=0; i<(end-begin); ++i)
		sparse_triplets.push_back(SparseTriplet(i,i,D(i)));

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	weight_matrix.cwiseMax(SparseWeightMatrix(weight_matrix.transpose()));

	return make_pair(weight_matrix,DenseDiagonalMatrix(D));
}

template<class RandomAccessIterator, class FeatureVectorCallback>
pair<DenseSymmetricMatrix,DenseSymmetricMatrix> construct_locality_preserving_eigenproblem(SparseWeightMatrix L,
		DenseDiagonalMatrix D, RandomAccessIterator begin, RandomAccessIterator end, FeatureVectorCallback feature_vector_callback,
		unsigned int dimension)
{
	timed_context context("Constructing LPP eigenproblem");

	DenseSymmetricMatrix lhs = DenseSymmetricMatrix::Zero(dimension,dimension);
	DenseSymmetricMatrix rhs = DenseSymmetricMatrix::Zero(dimension,dimension);

	DenseVector rank_update_vector_i(dimension);
	DenseVector rank_update_vector_j(dimension);
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback(*iter,rank_update_vector_i);
		rhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i,D.diagonal()(iter-begin));
	}

	for (int i=0; i<L.outerSize(); ++i)
	{
		for (SparseWeightMatrix::InnerIterator it(L,i); it; ++it)
		{
			feature_vector_callback(begin[it.row()],rank_update_vector_i);
			feature_vector_callback(begin[it.col()],rank_update_vector_j);
			lhs.selfadjointView<Eigen::Upper>().rankUpdate(rank_update_vector_i, rank_update_vector_j, it.value());
		}
	}

	return make_pair(lhs,rhs);
}

#endif
