/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (w) 2012, Fernando J. Iglesias Garcia
 * Copyright (c) 2012, Fernando J. Iglesias Garcia
 */

#ifndef TAPKEE_MVU_H_
#define TAPKEE_MVU_H_

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator>
Laplacian compute_laplacian(RandomAccessIterator begin, RandomAccessIterator end,
		const Neighbors& neighbors)
{
	timed_context("Laplacian computation");

	const unsigned int k = neighbors[0].size();
	SparseTriplets sparse_triplets;
	sparse_triplets.reserve(k*(end-begin));
	// Use DenseVector::Zero instead of DenseVector::Ones if a point is not
	// considered neighbour of itself
	DenseVector DD = DenseVector::Ones(end-begin);

	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		const LocalNeighbors& current_neighbors = neighbors[iter-begin];
		// Regard each point as neighbour of itself for the weight_matrix
		sparse_triplets.push_back(SparseTriplet(iter-begin,iter-begin,1));

		for (unsigned int i=0; i<k; ++i)
		{
			DD(iter-begin) += 1;
			sparse_triplets.push_back(SparseTriplet(iter-begin,current_neighbors[i],1));
		}
	}

	SparseWeightMatrix weight_matrix(end-begin,end-begin);
	weight_matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());

	return Laplacian(weight_matrix,DenseDiagonalMatrix(DD));
}

//TODO
template <class RandomAccessIterator, class PairwiseCallback>
EmbeddingResult mvu_embedding(RandomAccessIterator begin, RandomAccessIterator end,
		PairwiseCallback callback, const Neighbors& neighbors,
		unsigned int target_dimension, TAPKEE_EIGEN_EMBEDDING_METHOD eigen_method)
{
	Laplacian laplacian = compute_laplacian(begin,end,neighbors);

	return EmbeddingResult(DenseMatrix(),DenseVector());
}

}
}

#endif /* TAPKEE_MVU_H_ */
