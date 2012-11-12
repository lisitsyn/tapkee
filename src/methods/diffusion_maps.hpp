/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_DIFFUSION_MAPS_H_
#define TAPKEE_DIFFUSION_MAPS_H_

#include "../defines.hpp"
#include "../utils/time.hpp"

template <class RandomAccessIterator, class PairwiseCallback>
DenseSymmetricMatrix compute_diffusion_matrix(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback callback, 
                                              unsigned int timesteps, DefaultScalarType width)
{
	timed_context context("Diffusion map matrix computation");

	DenseSymmetricMatrix diffusion_matrix(end-begin,end-begin);
	
	DenseVector p = DenseVector::Zero(end-begin);
	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=i_iter; j_iter!=end; ++j_iter)
		{
			DefaultScalarType k = callback(*i_iter,*j_iter);
			DefaultScalarType gk = exp(-(k*k)/width);
			diffusion_matrix(i_iter-begin,j_iter-begin) = gk;
			diffusion_matrix(j_iter-begin,i_iter-begin) = gk;
		}
	}
	p = diffusion_matrix.colwise().sum();

	for (unsigned int i=0; i<(end-begin); i++)
	{
		for (unsigned int j=0; j<(end-begin); j++)
			diffusion_matrix(i,j) /= pow(p(i)*p(j),timesteps);
	}

	p = diffusion_matrix.colwise().sum().cwiseSqrt();
	
	for (unsigned int i=0; i<(end-begin); i++)
	{
		for (unsigned int j=0; j<(end-begin); j++)
			diffusion_matrix(i,j) /= p(i)*p(j);
	}

	return diffusion_matrix;
};

#endif
