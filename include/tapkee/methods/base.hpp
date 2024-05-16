
/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/neighbors/neighbors.hpp>
#include <tapkee/parameters/context.hpp>
#include <tapkee/parameters/defaults.hpp>
#include <tapkee/predicates.hpp>
#include <tapkee/projection.hpp>
#include <tapkee/utils/features.hpp>
#include <tapkee/utils/logging.hpp>
#include <tapkee/utils/naming.hpp>
#include <tapkee/utils/time.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>
class ImplementationBase
{
  public:
    ImplementationBase(RandomAccessIterator b, RandomAccessIterator e, KernelCallback k, DistanceCallback d,
                       FeaturesCallback f, ParametersSet& pmap, const Context& ctx)
        : parameters(pmap), context(ctx), kernel(k), distance(d), features(f),
          plain_distance(PlainDistance<RandomAccessIterator, DistanceCallback>(distance)),
          kernel_distance(KernelDistance<RandomAccessIterator, KernelCallback>(kernel)), begin(b), end(e), n_vectors(0),
          current_dimension(0)
    {
        n_vectors = (end - begin);

        if (n_vectors == 0)
            throw no_data_error();

        parameters[target_dimension].checked().satisfies(InRange<IndexType>(1, n_vectors)).orThrow();

        if (!is_dummy<FeaturesCallback>::value)
            current_dimension = features.dimension();
        else
            current_dimension = 0;
    }
    ImplementationBase(const ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>& other)
        : parameters(other.parameters)
        , context(other.context)
        , kernel(other.kernel)
        , distance(other.distance)
        , features(other.features)
        , plain_distance(other.plain_distance)
        , kernel_distance(other.kernel_distance)
        , begin(other.begin)
        , end(other.end)
        , n_vectors(other.n_vectors)
        , current_dimension(other.current_dimension)
    {

    }

  protected:
    ParametersSet parameters;
    Context context;
    KernelCallback kernel;
    DistanceCallback distance;
    FeaturesCallback features;
    PlainDistance<RandomAccessIterator, DistanceCallback> plain_distance;
    KernelDistance<RandomAccessIterator, KernelCallback> kernel_distance;

    RandomAccessIterator begin;
    RandomAccessIterator end;

    IndexType n_vectors;
    IndexType current_dimension;

  protected:
    template <class Distance>
    Neighbors find_neighbors_with(Distance d)
    {
        parameters[num_neighbors].checked().satisfies(InRange<IndexType>(3, n_vectors)).orThrow();
        return find_neighbors(parameters[neighbors_method], begin, end, d, parameters[num_neighbors],
                              parameters[check_connectivity]);
    }

    template <class MatrixType>
    EigendecompositionResult eigendecomposition_via(const EigendecompositionStrategy& eigen_strategy, const MatrixType& m, IndexType target_dimension)
    {
        return eigendecomposition(
            parameters[eigen_method],
            parameters[computation_strategy],
            eigen_strategy,
            m,
            target_dimension
        );
    }


};

// TODO can we avoid these using things?
#define __TAPKEE_IMPLEMENTATION(Method)                                                                                                           \
    template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>                                   \
    class Method ## Implementation : public ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>          \
    {                                                                                                                                             \
    public:                                                                                                                                       \
        typedef ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback> Base;                                \
        using Base::parameters;                                                                                                                   \
        using Base::context;                                                                                                                      \
        using Base::kernel;                                                                                                                       \
        using Base::distance;                                                                                                                     \
        using Base::features;                                                                                                                     \
        using Base::plain_distance;                                                                                                               \
        using Base::kernel_distance;                                                                                                              \
        using Base::begin;                                                                                                                        \
        using Base::end;                                                                                                                          \
        using Base::n_vectors;                                                                                                                    \
        using Base::current_dimension;                                                                                                            \
        using Base::find_neighbors_with;                                                                                                          \
        using Base::eigendecomposition_via;                                                                                                       \
        Method ## Implementation(const ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>& other) :     \
            ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>(other)                                   \
        {                                                                                                                                         \
        }
#define __TAPKEE_END_IMPLEMENTATION() };

} // End of namespace tapkee_internal
} // End of namespace tapkee
