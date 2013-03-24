/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_MAIN_H_
#define TAPKEE_MAIN_H_

/* Tapkee includes */
#include <tapkee_defines.hpp>
#include <tapkee_methods.hpp>
#include <callback/eigen_callbacks.hpp>
/* End of Tapkee includes */

namespace tapkee
{

/** Main entry-point of the library. Constructs a dense embedding with specified 
 * dimensionality using provided data represented by random access iterators 
 * and provided callbacks. Returns ReturnType that is essentially a pair of 
 * DenseMatrix (embedding of provided data) and a ProjectingFunction with 
 * corresponding ProjectionImplementation used to project 
 * data out of the sample.
 *
 * @tparam RandomAccessIterator basic random access iterator with no specific capabilities.
 *
 * @tparam KernelCallback that defines 
 * @code ScalarType operator()( RandomAccessIterator, RandomAccessIterator) @endcode 
 * function of two iterators. This method should return value of Mercer kernel function 
 * between vectors/objects iterators pointing to. The callback should be marked as a kernel function using
 * @ref tapkee::TAPKEE_CALLBACK_IS_KERNEL macro (fails on compilation in other case).
 *
 * @tparam DistanceCallback that defines 
 * @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
 * function of two iterators. The callback should be marked as a distance function using 
 * @ref TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
 *
 * @tparam FeatureVectorCallback that defines 
 * @code void operator()(RandomAccessIterator, DenseVector) @endcode function
 * used to access feature vector pointed by iterator. The callback should put the feature vector 
 * pointed by iterator to the second argument vector.
 *
 * Parameters required by the chosen algorithm are obtained from the parameter map. It gracefully 
 * fails during runtime and throws an exception if some of required 
 * parameters are not specified or have improper values.
 *
 * @param begin begin iterator of data
 * @param end end iterator of data
 * @param kernel_callback the kernel callback implementing
 * @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
 *
 * Used by the following methods: 
 * - @ref tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING
 * - @ref tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING
 * - @ref tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT
 * - @ref tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT
 * - @ref tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING
 * - @ref tapkee::KERNEL_PCA
 *
 * @param distance_callback the distance callback implementing
 * @code ScalarType operator()(RandomAccessIterator, RandomAccessIterator) @endcode 
 *
 * Used by the following methods: 
 * - @ref tapkee::LAPLACIAN_EIGENMAPS
 * - @ref tapkee::LOCALITY_PRESERVING_PROJECTIONS
 * - @ref tapkee::DIFFUSION_MAP
 * - @ref tapkee::ISOMAP
 * - @ref tapkee::LANDMARK_ISOMAP
 * - @ref tapkee::MULTIDIMENSIONAL_SCALING
 * - @ref tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING
 * - @ref tapkee::STOCHASTIC_PROXIMITY_EMBEDDING
 * - @ref tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING
 *
 * @param feature_vector_callback the feature vector callback implementing
 * @code void operator()(RandomAccessIterator, DenseVector) @endcode
 *
 * Used by the following methods:
 * - @ref tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING
 * - @ref tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT
 * - @ref tapkee::LOCALITY_PRESERVING_PROJECTIONS
 * - @ref tapkee::PCA
 * - @ref tapkee::RANDOM_PROJECTION
 * - @ref tapkee::FACTOR_ANALYSIS
 * - @ref tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING
 * - @ref tapkee::PASS_THRU
 *
 * @param parameters parameter map containing values with keys from @ref tapkee::TAPKEE_PARAMETERS
 *
 * @throw @ref tapkee::wrong_parameter_error if wrong parameter value is passed into the parameters map
 * @throw @ref tapkee::wrong_parameter_type_error if a value with wrong type is passed into the parameters map
 * @throw @ref tapkee::missed_parameter_error if some required parameter is missed in the parameters map
 * @throw @ref tapkee::unsupported_method_error if some method or combination of methods is unsupported 
 * @throw @ref tapkee::not_enough_memory_error if there is not enough memory to perform the algorithm
 * @throw @ref tapkee::cancelled_exception if computations were cancelled with provided @ref tapkee::CANCEL_FUNCTION
 * @throw @ref tapkee::eigendecomposition_error if eigendecomposition is failed
 *
 */
template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
ReturnResult embed(RandomAccessIterator begin, RandomAccessIterator end,
                   KernelCallback kernel_callback, DistanceCallback distance_callback,
                   FeatureVectorCallback feature_vector_callback, ParametersMap parameters)
{
#if EIGEN_VERSION_AT_LEAST(3,1,0)
	Eigen::initParallel();
#endif
	ReturnResult return_result;

	TAPKEE_METHOD method;
	if (!parameters.count(REDUCTION_METHOD))
		throw missed_parameter_error("Dimension reduction wasn't specified");

	try 
	{
		method = parameters[REDUCTION_METHOD].cast<TAPKEE_METHOD>();
	}
	catch (const anyimpl::bad_any_cast&)
	{
		throw wrong_parameter_type_error("Wrong method type specified.");
	}

#define PUT_DEFAULT(KEY,TYPE,VALUE)                 \
	if (!parameters.count(KEY))                     \
		parameters[KEY] = static_cast<TYPE>(VALUE); 

	//// defaults
	PUT_DEFAULT(OUTPUT_FEATURE_VECTORS_ARE_COLUMNS,bool,false);
	PUT_DEFAULT(EIGENSHIFT,ScalarType,1e-9);
	PUT_DEFAULT(KLLE_TRACE_SHIFT,ScalarType,1e-3);
	PUT_DEFAULT(CHECK_CONNECTIVITY,bool,true);
	//// end of defaults

#undef PUT_DEFAULT

	void (*progress_function)(double) = NULL;
	bool (*cancel_function)() = NULL;

	if (parameters.count(PROGRESS_FUNCTION))
		progress_function = parameters[PROGRESS_FUNCTION].cast<void (*)(double)>();
	if (parameters.count(CANCEL_FUNCTION))
		cancel_function = parameters[CANCEL_FUNCTION].cast<bool (*)()>();

	tapkee_internal::Context context(progress_function,cancel_function);

#define CALL_IMPLEMENTATION(X)                                                                                              \
		tapkee_internal::Implementation<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback,X>()     \
		(begin,end,tapkee_internal::Callbacks<KernelCallback,DistanceCallback,FeatureVectorCallback>(kernel_callback,       \
		distance_callback,feature_vector_callback),parameters,context)
#define HANDLE_IMPLEMENTATION(X)                          \
	case X: return_result = CALL_IMPLEMENTATION(X); break

	try 
	{
		LoggingSingleton::instance().message_info("Using " + tapkee_internal::get_method_name(method) + " method.");
		switch (method)
		{
			HANDLE_IMPLEMENTATION(KERNEL_LOCALLY_LINEAR_EMBEDDING);
			HANDLE_IMPLEMENTATION(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT);
			HANDLE_IMPLEMENTATION(DIFFUSION_MAP);
			HANDLE_IMPLEMENTATION(MULTIDIMENSIONAL_SCALING);
			HANDLE_IMPLEMENTATION(LANDMARK_MULTIDIMENSIONAL_SCALING);
			HANDLE_IMPLEMENTATION(ISOMAP);
			HANDLE_IMPLEMENTATION(LANDMARK_ISOMAP);
			HANDLE_IMPLEMENTATION(NEIGHBORHOOD_PRESERVING_EMBEDDING);
			HANDLE_IMPLEMENTATION(LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
			HANDLE_IMPLEMENTATION(HESSIAN_LOCALLY_LINEAR_EMBEDDING);
			HANDLE_IMPLEMENTATION(LAPLACIAN_EIGENMAPS);
			HANDLE_IMPLEMENTATION(LOCALITY_PRESERVING_PROJECTIONS);
			HANDLE_IMPLEMENTATION(PCA);
			HANDLE_IMPLEMENTATION(KERNEL_PCA);
			HANDLE_IMPLEMENTATION(RANDOM_PROJECTION);
			HANDLE_IMPLEMENTATION(STOCHASTIC_PROXIMITY_EMBEDDING);
			HANDLE_IMPLEMENTATION(PASS_THRU);
			HANDLE_IMPLEMENTATION(FACTOR_ANALYSIS);
			HANDLE_IMPLEMENTATION(T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING);
			case UNKNOWN_METHOD: throw wrong_parameter_error("unknown method"); break;
		}
	}
	catch (const std::bad_alloc&)
	{
		LoggingSingleton::instance().message_error("Not enough memory available.");
		throw not_enough_memory_error("Not enough memory");
	}

#undef CALL_IMPLEMENTATION 
#undef HANDLE_IMPLEMENTATION

	if (parameters[OUTPUT_FEATURE_VECTORS_ARE_COLUMNS].cast<bool>())
		return_result.first.transposeInPlace();

	return return_result;
}

/** Additional entry-point of the library. Constructs a dense embedding with specified 
 * dimensionality using provided number of vectors and callbacks. This function 
 * enumerates vectors with 0..N-1 values and use these indices with provided 
 * callbacks. Returns ReturnType that is essentially a pair of a DenseMatrix 
 * (embedding of provided data) and a ProjectingFunction with corresponding 
 * ProjectionImplementation which can be used to project data out of the sample.
 *
 * @tparam KernelCallback that defines 
 * @code ScalarType operator()(IndexType, IndexType) @endcode 
 * function of two indices in range 0..N-1. This method should return value of Mercer kernel function 
 * between vectors/objects with specified indices. The callback should be marked as a
 * kernel function using @ref tapkee::TAPKEE_CALLBACK_IS_KERNEL 
 * macro (fails on compilation in other case).
 *
 * @tparam DistanceCallback that defines 
 * @code ScalarType operator()(IndexType, IndexType) @endcode 
 * function of two indices in range 0..N-1. The callback should be marked as a 
 * distance function using @ref TAPKEE_CALLBACK_IS_DISTANCE macro 
 * (fails during compilation in other case).
 *
 * @tparam FeatureVectorCallback that defines 
 * @code void operator()(IndexType, DenseVector) @endcode function
 * used to access feature vector pointed by its index. The callback should put the feature vector 
 * with the index to the second argument vector.
 *
 * Parameters required by the chosen algorithm are obtained from the parameter map. It gracefully 
 * fails during runtime and throws an exception if some of required 
 * parameters are not specified or have improper values.
 *
 * @param N total number of vectors
 * @param kernel_callback the kernel callback implementing
 * @code ScalarType operator()(IndexType, IndexType) @endcode 
 *
 * Used by the following methods: 
 * - @ref tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING
 * - @ref tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING
 * - @ref tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT
 * - @ref tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT
 * - @ref tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING
 * - @ref tapkee::KERNEL_PCA
 *
 * @param distance_callback the distance callback implementing
 * @code ScalarType operator()(IndexType, RandomAccessIterator) @endcode 
 *
 * Used by the following methods: 
 * - @ref tapkee::LAPLACIAN_EIGENMAPS
 * - @ref tapkee::LOCALITY_PRESERVING_PROJECTIONS
 * - @ref tapkee::DIFFUSION_MAP
 * - @ref tapkee::ISOMAP
 * - @ref tapkee::LANDMARK_ISOMAP
 * - @ref tapkee::MULTIDIMENSIONAL_SCALING
 * - @ref tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING
 * - @ref tapkee::STOCHASTIC_PROXIMITY_EMBEDDING
 * - @ref tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING
 *
 * @param feature_vector_callback the feature vector callback implementing
 * @code void operator()(IndexType, DenseVector) @endcode
 *
 * Used by the following methods:
 * - @ref tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING
 * - @ref tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT
 * - @ref tapkee::LOCALITY_PRESERVING_PROJECTIONS
 * - @ref tapkee::PCA
 * - @ref tapkee::RANDOM_PROJECTION
 * - @ref tapkee::FACTOR_ANALYSIS
 * - @ref tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING
 * - @ref tapkee::PASS_THRU
 *
 * @param parameters parameter map containing values with keys from @ref tapkee::TAPKEE_PARAMETERS
 *
 * @throw @ref tapkee::wrong_parameter_error if wrong parameter value is passed into the parameters map
 * @throw @ref tapkee::wrong_parameter_type_error if a value with wrong type is passed into the parameters map
 * @throw @ref tapkee::missed_parameter_error if some required parameter is missed in the parameters map
 * @throw @ref tapkee::unsupported_method_error if some method or combination of methods is unsupported 
 * @throw @ref tapkee::not_enough_memory_error if there is not enough memory to perform the algorithm
 * @throw @ref tapkee::cancelled_exception if computations were cancelled with provided @ref tapkee::CANCEL_FUNCTION
 * @throw @ref tapkee::eigendecomposition_error if eigendecomposition is failed
 *
 */
template <class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
ReturnResult embed(IndexType N, KernelCallback kernel_callback, DistanceCallback distance_callback,
                   FeatureVectorCallback feature_vector_callback, ParametersMap parameters)
{
	tapkee_internal::VectorIndices indices(N);
	for (IndexType i=0; i<N; ++i) 
		indices[i] = i;

	return embed(indices.begin(),indices.end(),kernel_callback,distance_callback,
	             feature_vector_callback,parameters);
}

/** Additional simplified entry-point of the library. Constructs 
 * a dense embedding with specified dimensionality using provided data matrix.
 * Returns ReturnType that is essentially a pair of a DenseMatrix 
 * (embedding of provided data) and a ProjectingFunction with corresponding 
 * ProjectionImplementation which can be used to project data out of the sample.
 *
 * This function uses euclidean distance and linear kernel as callbacks.
 * 
 * @param data data matrix (column-wise, i.e. each vector to be projected 
 *        is a column of that matrix)
 * @param parameters parameter map containing values with keys from @ref tapkee::TAPKEE_PARAMETERS
 *
 * @throw @ref tapkee::wrong_parameter_error if wrong parameter value is passed into the parameters map
 * @throw @ref tapkee::wrong_parameter_type_error if a value with wrong type is passed into the parameters map
 * @throw @ref tapkee::missed_parameter_error if some required parameter is missed in the parameters map
 * @throw @ref tapkee::unsupported_method_error if some method or combination of methods is unsupported 
 * @throw @ref tapkee::not_enough_memory_error if there is not enough memory to perform the algorithm
 * @throw @ref tapkee::cancelled_exception if computations were cancelled with provided @ref tapkee::CANCEL_FUNCTION
 * @throw @ref tapkee::eigendecomposition_error if eigendecomposition is failed
 *
 */
ReturnResult embed(const DenseMatrix& data, ParametersMap parameters)
{
	distance_callback dcb(data);
	kernel_callback kcb(data);
	feature_vector_callback fcb(data);
	parameters[CURRENT_DIMENSION] = static_cast<IndexType>(data.rows());
	ReturnResult result = embed(static_cast<IndexType>(data.cols()),kcb,dcb,fcb,parameters);
	return result;
}

} // End of namespace tapkee

#endif
