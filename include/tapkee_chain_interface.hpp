/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_CHAIN_H_
#define TAPKEE_CHAIN_H_

#include <tapkee_embed.hpp>
#include <callbacks/dummy_callbacks.hpp>
#include <callbacks/eigen_callbacks.hpp>

namespace tapkee
{

namespace tapkee_internal
{
	template<class KernelCallback, class DistanceCallback, class FeaturesCallback>
	class CallbacksInitializedState
	{
	public:
		CallbacksInitializedState(const ParametersSet& params, const KernelCallback& k,
								  const DistanceCallback& d, const FeaturesCallback& f) :
			parameters(params), kernel(k), distance(d), features(f) { }
		
		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return tapkee::embed(begin,end,kernel,distance,features,parameters);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		KernelCallback kernel;
		DistanceCallback distance;
		FeaturesCallback features;
	};

	template<class KernelCallback, class DistanceCallback>
	class KernelAndDistanceInitializedState
	{
	public:
		KernelAndDistanceInitializedState(const ParametersSet& params, const KernelCallback& k, const DistanceCallback& d) :
			parameters(params), kernel(k), distance(d) { }

		template<class FeaturesCallback>
		CallbacksInitializedState<KernelCallback,DistanceCallback,FeaturesCallback> 
			withFeatures(const FeaturesCallback& features) const
		{ return CallbacksInitializedState<KernelCallback,DistanceCallback,FeaturesCallback>(parameters,kernel,distance,features); }
		
		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return (*this).withFeatures(dummy_features_callback<typename RandomAccessIterator::value_type>())
						  .embedRange(begin,end);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		KernelCallback kernel;
		DistanceCallback distance;
	};

	template<class KernelCallback, class FeaturesCallback>
	class KernelAndFeaturesInitializedState
	{
	public:
		KernelAndFeaturesInitializedState(const ParametersSet& params, const KernelCallback& k, const FeaturesCallback& f) :
			parameters(params), kernel(k), features(f) { }

		template<class DistanceCallback>
		CallbacksInitializedState<KernelCallback,DistanceCallback,FeaturesCallback> 
			withDistance(const DistanceCallback& distance) const
		{ return CallbacksInitializedState<KernelCallback,DistanceCallback,FeaturesCallback>(parameters,kernel,distance,features); }

		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return (*this).withDistance(dummy_distance_callback<typename RandomAccessIterator::value_type>())
						  .embedRange(begin,end);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		KernelCallback kernel;
		FeaturesCallback features;
	};

	template<class DistanceCallback, class FeaturesCallback>
	class DistanceAndFeaturesInitializedState
	{
	public:
		DistanceAndFeaturesInitializedState(const ParametersSet& params, const DistanceCallback& d, const FeaturesCallback& f) :
			parameters(params), distance(d), features(f) { }

		template<class KernelCallback>
		CallbacksInitializedState<KernelCallback,DistanceCallback,FeaturesCallback> 
			withKernel(const KernelCallback& kernel) const
		{ return CallbacksInitializedState<KernelCallback,DistanceCallback,FeaturesCallback>(parameters,kernel,distance,features); }

		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return (*this).withKernel(dummy_kernel_callback<typename RandomAccessIterator::value_type>())
						  .embedRange(begin,end);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		DistanceCallback distance;
		FeaturesCallback features;
	};

	template<class KernelCallback>
	class KernelFirstInitializedState
	{
	public:
		KernelFirstInitializedState(const ParametersSet& params, const KernelCallback& callback) :
			parameters(params), kernel(callback) { }

		template<class DistanceCallback>
		KernelAndDistanceInitializedState<KernelCallback,DistanceCallback> withDistance(const DistanceCallback& distance) const
		{ return KernelAndDistanceInitializedState<KernelCallback,DistanceCallback>(parameters,kernel,distance); }
		template<class FeaturesCallback>
		KernelAndFeaturesInitializedState<KernelCallback,FeaturesCallback> withFeatures(const FeaturesCallback& features) const
		{ return KernelAndFeaturesInitializedState<KernelCallback,FeaturesCallback>(parameters,kernel,features); }

		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return (*this).withDistance(dummy_distance_callback<typename RandomAccessIterator::value_type>())
						  .withFeatures(dummy_features_callback<typename RandomAccessIterator::value_type>())
						  .embedRange(begin,end);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		KernelCallback kernel;
	};

	template<class DistanceCallback>
	class DistanceFirstInitializedState
	{
	public:
		DistanceFirstInitializedState(const ParametersSet& params, const DistanceCallback& callback) :
			parameters(params), distance(callback) { }

		template<class KernelCallback>
		KernelAndDistanceInitializedState<KernelCallback,DistanceCallback> withKernel(const KernelCallback& kernel) const
		{ return KernelAndDistanceInitializedState<KernelCallback,DistanceCallback>(parameters,kernel,distance); }
		template<class FeaturesCallback>
		DistanceAndFeaturesInitializedState<DistanceCallback,FeaturesCallback> withFeatures(const FeaturesCallback& features) const
		{ return DistanceAndFeaturesInitializedState<DistanceCallback,FeaturesCallback>(parameters,distance,features); }

		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return (*this).withKernel(dummy_kernel_callback<typename RandomAccessIterator::value_type>())
						  .withFeatures(dummy_features_callback<typename RandomAccessIterator::value_type>())
						  .embedRange(begin,end);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		DistanceCallback distance;
	};

	template<class FeaturesCallback>
	class FeaturesFirstInitializedState
	{
	public:
		FeaturesFirstInitializedState(const ParametersSet& params, const FeaturesCallback& callback) :
			parameters(params), features(callback) { }

		template<class KernelCallback>
		KernelAndFeaturesInitializedState<KernelCallback,FeaturesCallback> withKernel(const KernelCallback& kernel) const
		{ return KernelAndFeaturesInitializedState<KernelCallback,FeaturesCallback>(parameters,kernel,features); }
		template<class DistanceCallback>
		DistanceAndFeaturesInitializedState<DistanceCallback,FeaturesCallback> withDistance(const DistanceCallback& distance) const
		{ return DistanceAndFeaturesInitializedState<DistanceCallback,FeaturesCallback>(parameters,distance,features); }

		template<class RandomAccessIterator>
		TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
		{
			return (*this).withKernel(dummy_kernel_callback<typename RandomAccessIterator::value_type>())
						  .withDistance(dummy_distance_callback<typename RandomAccessIterator::value_type>())
						  .embedRange(begin,end);
		}
		template<class Container>
		TapkeeOutput embedUsing(const Container& container) const
		{
			return embedRange(container.begin(),container.end());
		}
	private:
		ParametersSet parameters;
		FeaturesCallback features;
	};

	class ParametersInitializedState
	{
	public:
		ParametersInitializedState(const ParametersSet& that) : parameters(that) { }
		ParametersInitializedState(const ParametersInitializedState&);
		ParametersInitializedState& operator=(const ParametersInitializedState&);
		template<class KernelCallback> 
		KernelFirstInitializedState<KernelCallback> withKernel(const KernelCallback& callback) const
		{ return KernelFirstInitializedState<KernelCallback>(parameters,callback); }
		template<class DistanceCallback>
		DistanceFirstInitializedState<DistanceCallback> withDistance(const DistanceCallback& callback) const
		{ return DistanceFirstInitializedState<DistanceCallback>(parameters,callback); }
		template<class FeaturesCallback>
		FeaturesFirstInitializedState<FeaturesCallback> withFeatures(const FeaturesCallback& features) const
		{ return FeaturesFirstInitializedState<FeaturesCallback>(parameters,features); }

		TapkeeOutput embedUsing(const DenseMatrix& matrix) const
		{
			vector<int> indices(matrix.cols());
			for (int i=0; i<matrix.cols(); i++) indices[i] = i;
			eigen_kernel_callback kcb(matrix);
			eigen_distance_callback dcb(matrix);
			eigen_features_callback fvcb(matrix);
			return tapkee::embed(indices.begin(),indices.end(),kcb,dcb,fvcb,parameters);
		}
	private:
		ParametersSet parameters;
	};
}

struct initialize
{
	initialize() 
	{
	}
	tapkee_internal::ParametersInitializedState operator[](const ParametersSet& parameters) const
	{
		return withParameters(parameters);
	}
	tapkee_internal::ParametersInitializedState withParameters(const ParametersSet& parameters) const
	{
		return tapkee_internal::ParametersInitializedState(parameters);
	}
};

}

#endif

