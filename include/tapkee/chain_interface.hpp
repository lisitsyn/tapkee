/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/callbacks/dummy_callbacks.hpp>
#include <tapkee/callbacks/eigen_callbacks.hpp>
#include <tapkee/embed.hpp>
/* End of Tapkee includes */

namespace tapkee
{

namespace tapkee_internal
{
template <class KernelCallback, class DistanceCallback, class FeaturesCallback> class CallbacksInitializedState
{
  public:
    CallbacksInitializedState(const ParametersSet& params, const KernelCallback& k, const DistanceCallback& d,
                              const FeaturesCallback& f)
        : parameters(params), kernel(k), distance(d), features(f)
    {
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return tapkee::embed(begin, end, kernel, distance, features, parameters);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    KernelCallback kernel;
    DistanceCallback distance;
    FeaturesCallback features;
};

template <class KernelCallback, class DistanceCallback> class KernelAndDistanceInitializedState
{
  public:
    KernelAndDistanceInitializedState(const ParametersSet& params, const KernelCallback& k, const DistanceCallback& d)
        : parameters(params), kernel(k), distance(d)
    {
    }

    /** Sets features callback.
     *
     * @param callback a callback that implements the
     *        @code vector(const RandomAccessIterator::value_type&, DenseVector&) @endcode
     *        member function which is used to obtain the feature vector pointed by the
     *        first argument.
     */
    template <class FeaturesCallback>
        requires requires(FeaturesCallback callback) { callback.dimension(); }
    CallbacksInitializedState<KernelCallback, DistanceCallback, FeaturesCallback> with(
        const FeaturesCallback& features) const
    {
        return CallbacksInitializedState<KernelCallback, DistanceCallback, FeaturesCallback>(parameters, kernel,
                                                                                             distance, features);
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return (*this)
            .with(dummy_features_callback<typename RandomAccessIterator::value_type>())
            .embedRange(begin, end);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    KernelCallback kernel;
    DistanceCallback distance;
};

template <class KernelCallback, class FeaturesCallback> class KernelAndFeaturesInitializedState
{
  public:
    KernelAndFeaturesInitializedState(const ParametersSet& params, const KernelCallback& k, const FeaturesCallback& f)
        : parameters(params), kernel(k), features(f)
    {
    }

    /** Sets distance callback.
     *
     * @param callback a callback that implements the
     *        @code distance(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute distance (dissimilarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class DistanceCallback>
        requires requires(DistanceCallback callback, ScalarType a, ScalarType b) { callback.distance(a, b); } ||
                 requires(DistanceCallback callback) { callback.distance_matrix; }
    CallbacksInitializedState<KernelCallback, DistanceCallback, FeaturesCallback> with(
        const DistanceCallback& distance) const
    {
        return CallbacksInitializedState<KernelCallback, DistanceCallback, FeaturesCallback>(parameters, kernel,
                                                                                             distance, features);
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return (*this)
            .with(dummy_distance_callback<typename RandomAccessIterator::value_type>())
            .embedRange(begin, end);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    KernelCallback kernel;
    FeaturesCallback features;
};

/**
 *
 *
 *
 */
template <class DistanceCallback, class FeaturesCallback> class DistanceAndFeaturesInitializedState
{
  public:
    DistanceAndFeaturesInitializedState(const ParametersSet& params, const DistanceCallback& d,
                                        const FeaturesCallback& f)
        : parameters(params), distance(d), features(f)
    {
    }

    /** Sets kernel callback.
     *
     * @param callback a callback that implements the
     *        @code kernel(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute kernel value (similarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class KernelCallback>
        requires requires(KernelCallback callback, ScalarType a, ScalarType b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback, const std::string& a, const std::string& b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback) { callback.kernel_matrix; }
    CallbacksInitializedState<KernelCallback, DistanceCallback, FeaturesCallback> with(
        const KernelCallback& kernel) const
    {
        return CallbacksInitializedState<KernelCallback, DistanceCallback, FeaturesCallback>(parameters, kernel,
                                                                                             distance, features);
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return (*this)
            .with(dummy_kernel_callback<typename RandomAccessIterator::value_type>())
            .embedRange(begin, end);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    DistanceCallback distance;
    FeaturesCallback features;
};

template <class KernelCallback> class KernelFirstInitializedState
{
  public:
    KernelFirstInitializedState(const ParametersSet& params, const KernelCallback& callback)
        : parameters(params), kernel(callback)
    {
    }

    /** Sets distance callback.
     *
     * @param callback a callback that implements the
     *        @code distance(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute distance (dissimilarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class DistanceCallback>
        requires requires(DistanceCallback callback, ScalarType a, ScalarType b) { callback.distance(a, b); } ||
                 requires(DistanceCallback callback) { callback.distance_matrix; }
    KernelAndDistanceInitializedState<KernelCallback, DistanceCallback> with(
        const DistanceCallback& callback) const
    {
        return KernelAndDistanceInitializedState<KernelCallback, DistanceCallback>(parameters, kernel, callback);
    }

    /** Sets features callback.
     *
     * @param callback a callback that implements the
     *        @code vector(const RandomAccessIterator::value_type&, DenseVector&) @endcode
     *        member function which is used to obtain the feature vector pointed by the
     *        first argument.
     */
    template <class FeaturesCallback>
        requires requires(FeaturesCallback callback) { callback.dimension(); }
    KernelAndFeaturesInitializedState<KernelCallback, FeaturesCallback> with(
        const FeaturesCallback& callback) const
    {
        return KernelAndFeaturesInitializedState<KernelCallback, FeaturesCallback>(parameters, kernel, callback);
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return (*this)
            .with(dummy_distance_callback<typename RandomAccessIterator::value_type>())
            .with(dummy_features_callback<typename RandomAccessIterator::value_type>())
            .embedRange(begin, end);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    KernelCallback kernel;
};

template <class DistanceCallback> class DistanceFirstInitializedState
{
  public:
    DistanceFirstInitializedState(const ParametersSet& params, const DistanceCallback& callback)
        : parameters(params), distance(callback)
    {
    }

    /** Sets kernel callback.
     *
     * @param callback a callback that implements the
     *        @code kernel(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute kernel value (similarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class KernelCallback>
        requires requires(KernelCallback callback, ScalarType a, ScalarType b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback, const std::string& a, const std::string& b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback) { callback.kernel_matrix; }
    KernelAndDistanceInitializedState<KernelCallback, DistanceCallback> with(const KernelCallback& callback) const
    {
        return KernelAndDistanceInitializedState<KernelCallback, DistanceCallback>(parameters, callback, distance);
    }

    /** Sets features callback.
     *
     * @param callback a callback that implements the
     *        @code vector(const RandomAccessIterator::value_type&, DenseVector&) @endcode
     *        member function which is used to obtain the feature vector pointed by the
     *        first argument.
     */
    template <class FeaturesCallback>
        requires requires(FeaturesCallback callback) { callback.dimension(); }
    DistanceAndFeaturesInitializedState<DistanceCallback, FeaturesCallback> with(
        const FeaturesCallback& callback) const
    {
        return DistanceAndFeaturesInitializedState<DistanceCallback, FeaturesCallback>(parameters, distance, callback);
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return (*this)
            .with(dummy_kernel_callback<typename RandomAccessIterator::value_type>())
            .with(dummy_features_callback<typename RandomAccessIterator::value_type>())
            .embedRange(begin, end);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    DistanceCallback distance;
};

template <class FeaturesCallback> class FeaturesFirstInitializedState
{
  public:
    FeaturesFirstInitializedState(const ParametersSet& params, const FeaturesCallback& callback)
        : parameters(params), features(callback)
    {
    }

    /** Sets kernel callback.
     *
     * @param callback a callback that implements the
     *        @code kernel(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute kernel value (similarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class KernelCallback>
        requires requires(KernelCallback callback, ScalarType a, ScalarType b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback, const std::string& a, const std::string& b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback) { callback.kernel_matrix; }
    KernelAndFeaturesInitializedState<KernelCallback, FeaturesCallback> with(const KernelCallback& callback) const
    {
        return KernelAndFeaturesInitializedState<KernelCallback, FeaturesCallback>(parameters, callback, features);
    }

    /** Sets distance callback.
     *
     * @param callback a callback that implements the
     *        @code distance(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute distance (dissimilarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class DistanceCallback>
        requires requires(DistanceCallback callback, ScalarType a, ScalarType b) { callback.distance(a, b); } ||
                 requires(DistanceCallback callback) { callback.distance_matrix; }
    DistanceAndFeaturesInitializedState<DistanceCallback, FeaturesCallback> with(
        const DistanceCallback& callback) const
    {
        return DistanceAndFeaturesInitializedState<DistanceCallback, FeaturesCallback>(parameters, callback, features);
    }

    /** Constructs an embedding using the data represented by the
     * begin and end iterators.
     *
     * @param begin an iterator that points to the beginning of data container
     * @param end an iterator that points to the end of data container
     */
    template <class RandomAccessIterator>
    TapkeeOutput embedRange(RandomAccessIterator begin, RandomAccessIterator end) const
    {
        return (*this)
            .with(dummy_kernel_callback<typename RandomAccessIterator::value_type>())
            .with(dummy_distance_callback<typename RandomAccessIterator::value_type>())
            .embedRange(begin, end);
    }

    /** Constructs an embedding using the data represented by the container.
     *
     * @param container a container that supports begin() and end() methods
     *                  to get corresponding iterators
     */
    template <class Container> TapkeeOutput embedUsing(const Container& container) const
    {
        return embedRange(container.begin(), container.end());
    }

  private:
    ParametersSet parameters;
    FeaturesCallback features;
};

class ParametersInitializedState
{
  public:
    ParametersInitializedState(const ParametersSet& that) : parameters(that)
    {
    }
    ParametersInitializedState(const ParametersInitializedState&);
    ParametersInitializedState& operator=(const ParametersInitializedState&);

    /** Sets kernel callback.
     *
     * @param callback a callback that implements the
     *        @code kernel(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute kernel value (similarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class KernelCallback>
        requires requires(KernelCallback callback, ScalarType a, ScalarType b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback, const std::string& a, const std::string& b) { callback.kernel(a, b); } ||
                 requires(KernelCallback callback) { callback.kernel_matrix; }
    KernelFirstInitializedState<KernelCallback> with(const KernelCallback& callback) const
    {
        return KernelFirstInitializedState<KernelCallback>(parameters, callback);
    }

    /** Sets distance callback.
     *
     * @param callback a callback that implements the
     *        @code distance(const RandomAccessIterator::value_type&, const RandomAccessIterator::value_type&) @endcode
     *        member function which is used to compute distance (dissimilarity) between two objects
     *        pointed by the first and the second arguments.
     */
    template <class DistanceCallback>
        requires requires(DistanceCallback callback, ScalarType a, ScalarType b) { callback.distance(a, b); } ||
                 requires(DistanceCallback callback) { callback.distance_matrix; }
    DistanceFirstInitializedState<DistanceCallback> with(const DistanceCallback& callback) const
    {
        return DistanceFirstInitializedState<DistanceCallback>(parameters, callback);
    }

    /** Sets features callback.
     *
     * @param callback a callback that implements the
     *        @code vector(const RandomAccessIterator::value_type&, DenseVector&) @endcode
     *        member function which is used to obtain the feature vector pointed by the
     *        first argument.
     */
    template <class FeaturesCallback>
        requires requires(FeaturesCallback callback) { callback.dimension(); }
    FeaturesFirstInitializedState<FeaturesCallback> with(const FeaturesCallback& callback) const
    {
        return FeaturesFirstInitializedState<FeaturesCallback>(parameters, callback);
    }

    /** Constructs an embedding using the data represented
     * by the feature matrix. Uses linear kernel (dot product)
     * and euclidean distance.
     *
     * @param matrix matrix that contains feature vectors column-wise
     */
    TapkeeOutput embedUsing(const DenseMatrix& matrix) const
    {
        std::vector<IndexType> indices(matrix.cols());
        for (IndexType i = 0; i < matrix.cols(); i++)
            indices[i] = i;
        eigen_kernel_callback kcb(matrix);
        eigen_distance_callback dcb(matrix);
        eigen_features_callback fcb(matrix);
        return tapkee::embed(indices.begin(), indices.end(), kcb, dcb, fcb, parameters);
    }

  private:
    ParametersSet parameters;
};
} /* End of namespace tapkee_internal */

/** Returns an instance representing a state with initialized parameters.
 *
 * In the chain this method's call is followed by any of
 * @ref tapkee_internal::ParametersInitializedState::embedUsing
 * @ref tapkee_internal::ParametersInitializedState::with(const KernelCallback&)
 * @ref tapkee_internal::ParametersInitializedState::with(const DistanceCallback&)
 * @ref tapkee_internal::ParametersInitializedState::with(const FeaturesCallback&)
 *
 * @param parameters a set of parameters formed by keywords assigned to values
 */
tapkee_internal::ParametersInitializedState with(const ParametersSet& parameters)
{
    return tapkee_internal::ParametersInitializedState(parameters);
}

} /* End of namespace tapkee */
