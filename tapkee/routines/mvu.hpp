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

//TODO
template <class RandomAccessIterator, class PairwiseCallback>
EmbeddingResult mvu_embedding(RandomAccessIterator begin, RandomAccessIterator end,
		PairwiseCallback callback, const Neighbors& neighbors,
		unsigned int target_dimension)
{
	std::cout << "MVU not yet implemented" << std::endl;
	return EmbeddingResult(DenseMatrix(),DenseVector());
}

}
}

#endif /* TAPKEE_MVU_H_ */
