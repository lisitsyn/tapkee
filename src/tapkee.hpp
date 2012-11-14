/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_MAIN_H_
#define TAPKEE_MAIN_H_

#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <algorithm>
#include <iostream>

#include "defines.hpp"
#include "tapkee_highlevel_methods.hpp"

/** Main entry-point of the library. Constructs dense embedding with specified dimension
 * using provided data and callbacks.
 *
 * Has four template parameters:
 * 
 * RandomAccessIterator basic random access iterator with no specific capabilities.
 * 
 * KernelCallback that defines DefaultScalarType operator()(RandomAccessIterator, RandomAccessIterator) operation 
 * between two iterators. The operation should return value of Mercer kernel function 
 * between vectors/objects iterators pointing to. KernelCallback should be marked as a kernel function using
 * TAPKEE_CALLBACK_IS_KERNEL macro (fails during compilation in other case).
 * 
 * DistanceCallback that defines DefaultScalarType operator()(RandomAccessIterator, RandomAccessIterator) operation
 * between two iterators. DistanceCallback should be marked as a distance function using 
 * TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
 * 
 * FeatureVectorCallback that defines void operator()(RandomAccessIterator, DenseVector) operation
 * used to access feature vector pointed by iterator. The callback should put the feature vector pointed by iterator
 * to the vector of second argument.
 *
 * Parameters required by the chosen algorithm are obtained from the parameter map. It fails during runtime if
 * some of required parameters are not specified or have improper values.
 *
 * @param begin begin iterator of data
 * @param end end iterator of data
 * @param kernel_callback the kernel callback described before
 * @param distance_callback the distance callback described before
 * @param feature_vector_callback the feature vector access callback descrbied before
 * @param options parameter map
 */
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
DenseMatrix embed(RandomAccessIterator begin, RandomAccessIterator end,
                  KernelCallback kernel_callback, DistanceCallback distance_callback,
                  FeatureVectorCallback feature_vector_callback, ParametersMap options)
{
	Eigen::initParallel();
	EmbeddingResult embedding_result;

	// load common parameters from the parameters map
	TAPKEE_METHOD method = 
		options[REDUCTION_METHOD].cast<TAPKEE_METHOD>();

#define CALL_IMPLEMENTATION(X) embedding_impl<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,X>().embed(\
		begin,end,kernel_callback,distance_callback,feature_vector_callback,options)
#define NO_IMPLEMENTATION_YET printf("Not implemented\n"); exit(EXIT_FAILURE)

	switch (method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			embedding_result = CALL_IMPLEMENTATION(KERNEL_LOCALLY_LINEAR_EMBEDDING); break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			embedding_result = CALL_IMPLEMENTATION(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT); break;
		case DIFFUSION_MAP:
			embedding_result = CALL_IMPLEMENTATION(DIFFUSION_MAP); break;
		case MULTIDIMENSIONAL_SCALING:
			embedding_result = CALL_IMPLEMENTATION(MULTIDIMENSIONAL_SCALING); break;
		case LANDMARK_MULTIDIMENSIONAL_SCALING:
			embedding_result = CALL_IMPLEMENTATION(LANDMARK_MULTIDIMENSIONAL_SCALING); break;
		case ISOMAP:
			embedding_result = CALL_IMPLEMENTATION(ISOMAP); break;
		case LANDMARK_ISOMAP:
			NO_IMPLEMENTATION_YET; break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			embedding_result = CALL_IMPLEMENTATION(NEIGHBORHOOD_PRESERVING_EMBEDDING); break;
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			embedding_result = CALL_IMPLEMENTATION(LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT); break;
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			embedding_result = CALL_IMPLEMENTATION(HESSIAN_LOCALLY_LINEAR_EMBEDDING); break;
		case LAPLACIAN_EIGENMAPS:
			embedding_result = CALL_IMPLEMENTATION(LAPLACIAN_EIGENMAPS); break;
		case LOCALITY_PRESERVING_PROJECTIONS:
			embedding_result = CALL_IMPLEMENTATION(LOCALITY_PRESERVING_PROJECTIONS); break;
		case PCA:
			embedding_result = CALL_IMPLEMENTATION(PCA); break;
		case KERNEL_PCA:
			embedding_result = CALL_IMPLEMENTATION(KERNEL_PCA); break;
		case STOCHASTIC_PROXIMITY_EMBEDDING:
			embedding_result = CALL_IMPLEMENTATION(STOCHASTIC_PROXIMITY_EMBEDDING); break;
		case MAXIMUM_VARIANCE_UNFOLDING:
			NO_IMPLEMENTATION_YET; break;
	}

#undef CALL_IMPLEMENTATION 
#undef NO_IMPLEMENTATION_YET

	return embedding_result.first;
}

#endif
