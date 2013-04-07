/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_MAIN_H_
#define TAPKEE_MAIN_H_

/* Tapkee includes */
#include <tapkee_defines.hpp>
#include <tapkee_methods.hpp>
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
 * @tparam @code RandomAccessIterator @endcode basic random access iterator with no 
 *         specific capabilities that points to some @code Data @endcode (the simplest
 *         case is @code Data = int @endcode).
 *
 * @tparam KernelCallback that defines 
 * @code ScalarType kernel(const Data&, const Data&) @endcode 
 * function of two iterators. This method should return value of Mercer kernel function 
 * between vectors/objects iterators pointing to. The callback should be marked as a kernel function using
 * @ref tapkee::TAPKEE_CALLBACK_IS_KERNEL macro (fails on compilation in other case).
 *
 * @tparam DistanceCallback that defines 
 * @code ScalarType distance(const Data&, const Data&) @endcode 
 * function of two iterators. The callback should be marked as a distance function using 
 * @ref tapkee::TAPKEE_CALLBACK_IS_DISTANCE macro (fails during compilation in other case).
 *
 * @tparam FeatureVectorCallback that defines 
 * @code void vector(const Data&, DenseVector&) @endcode function
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
 * @code ScalarType kernel(const Data&, const Data&) @endcode 
 *
 * Used by the following methods: 
 * - @ref tapkee::KernelLocallyLinearEmbedding
 * - @ref tapkee::NeighborhoodPreservingEmbedding
 * - @ref tapkee::KernelLocalTangentSpaceAlignment
 * - @ref tapkee::LinearLocalTangentSpaceAlignment
 * - @ref tapkee::HessianLocallyLinearEmbedding
 * - @ref tapkee::KernelPCA
 *
 * @param distance_callback the distance callback implementing
 * @code ScalarType distance(const Data&, const Data&) @endcode 
 *
 * Used by the following methods: 
 * - @ref tapkee::LaplacianEigenmaps
 * - @ref tapkee::LocalityPreservingProjections
 * - @ref tapkee::DiffusionMap
 * - @ref tapkee::Isomap
 * - @ref tapkee::LandmarkIsomap
 * - @ref tapkee::MultidimensionalScaling
 * - @ref tapkee::LandmarkMultidimensionalScaling
 * - @ref tapkee::StochasticProximityEmbedding
 * - @ref tapkee::tDistributedStochasticNeighborEmbedding
 *
 * @param feature_vector_callback the feature vector callback implementing
 * @code void vector(const Data&, DenseVector&) @endcode
 *
 * Used by the following methods:
 * - @ref tapkee::NeighborhoodPreservingEmbedding
 * - @ref tapkee::LinearLocalTangentSpaceAlignment
 * - @ref tapkee::LocalityPreservingProjections
 * - @ref tapkee::PCA
 * - @ref tapkee::RandomProjection
 * - @ref tapkee::FactorAnalysis
 * - @ref tapkee::tDistributedStochasticNeighborEmbedding
 * - @ref tapkee::PassThru
 *
 * @param parameters parameter map containing values with keys from @ref tapkee::TAPKEE_PARAMETERS
 *
 * @throw @ref tapkee::wrong_parameter_error if wrong parameter value is passed into the parameters map
 * @throw @ref tapkee::wrong_parameter_type_error if a value with wrong type is passed into the parameters map
 * @throw @ref tapkee::missed_parameter_error if some required parameter is missed in the parameters map
 * @throw @ref tapkee::unsupported_method_error if some method or combination of methods is unsupported 
 * @throw @ref tapkee::not_enough_memory_error if there is not enough memory to perform the algorithm
 * @throw @ref tapkee::cancelled_exception if computations were cancelled with provided @ref tapkee::CANCEL_FUNCTION
 * @throw @ref tapkee::eigendecomposition_error if eigendecomposition has failed
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

	MethodId method;
	if (!parameters.count(ReductionMethod))
		throw missed_parameter_error("Dimension reduction method wasn't specified");

	try 
	{
		method = parameters[ReductionMethod];
	}
	catch (const anyimpl::bad_any_cast&)
	{
		throw wrong_parameter_type_error("Wrong method type specified.");
	}

#define PUT_DEFAULT(KEY,TYPE,VALUE)                 \
	if (!parameters.count(KEY))                     \
		parameters[KEY] = static_cast<TYPE>(VALUE); 

	//// defaults
	PUT_DEFAULT(NullspaceShift,ScalarType,1e-9);
	PUT_DEFAULT(KlleShift,ScalarType,1e-3);
	PUT_DEFAULT(CheckConnectivity,bool,true);
	PUT_DEFAULT(LandmarkRatio,ScalarType,0.5);

#ifdef TAPKEE_WITH_ARPACK
	PUT_DEFAULT(EigenEmbeddingMethod,EigenEmbeddingMethodId,Arpack);
#else
	PUT_DEFAULT(EigenEmbeddingMethod,EigenEmbeddingMethodId,Dense);
#endif

	PUT_DEFAULT(TargetDimension,IndexType,2);
#ifdef TAPKEE_USE_LGPL_COVERTREE
	PUT_DEFAULT(NeighborsMethod,NeighborsMethodId,CoverTree);
#endif
	//// end of defaults

#undef PUT_DEFAULT

	void (*progress_function)(double) = NULL;
	bool (*cancel_function)() = NULL;

	if (parameters.count(ProgressFunction))
		progress_function = parameters[ProgressFunction].cast<void (*)(double)>();
	if (parameters.count(CancelFunction))
		cancel_function = parameters[CancelFunction].cast<bool (*)()>();

	tapkee_internal::Context context(progress_function,cancel_function);

#define CALL_IMPLEMENTATION(X)                                                                                          \
		tapkee_internal::X##Implementation<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback> \
		(begin,end,tapkee_internal::Callbacks<KernelCallback,DistanceCallback,FeatureVectorCallback>(kernel_callback,   \
		distance_callback,feature_vector_callback),parameters,context)
#define HANDLE_IMPLEMENTATION(X)                          \
	case X: return_result = CALL_IMPLEMENTATION(X); break

	try 
	{
		LoggingSingleton::instance().message_info("Using the " + get_method_name(method) + " method.");
		
		return_result = 
			tapkee_internal::initialize(begin,end,kernel_callback,distance_callback,feature_vector_callback,parameters,context).embed(method);
	}
	catch (const std::bad_alloc&)
	{
		throw not_enough_memory_error("Not enough memory");
	}

#undef CALL_IMPLEMENTATION 
#undef HANDLE_IMPLEMENTATION

	return return_result;
}

} // End of namespace tapkee

#endif
