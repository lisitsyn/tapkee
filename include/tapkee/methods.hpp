/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/parameters/context.hpp>
#include <tapkee/parameters/defaults.hpp>
#include <tapkee/methods/all.hpp>
/* End of Tapkee includes */

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{

template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>
class DynamicImplementation : public ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>
{
public:
    using ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>::ImplementationBase;
    TapkeeOutput embedUsing(const DimensionReductionMethod& method)
    {
        timed_context tctx__(fmt::format("[+] embedding with {}", method.name()));

        if (this->context.is_cancelled())
            throw cancelled_exception();

        if (method.needs_kernel && is_dummy<KernelCallback>::value)
        {
            throw unsupported_method_error("Kernel callback is missed");
        }
        if (method.needs_distance && is_dummy<DistanceCallback>::value)
        {
            throw unsupported_method_error("Distance callback is missed");
        }
        if (method.needs_features && is_dummy<FeaturesCallback>::value)
        {
            throw unsupported_method_error("Features callback is missed");
        }

        const auto& self = static_cast<ImplementationBase<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>>(*this);

#define tapkee_method_handle(X)                                                                                        \
    if (method == X) {                                                                                                 \
        auto implementation =                                                                                          \
                X ## Implementation<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>(self);   \
        return implementation.embed();                                                                                 \
    }
        tapkee_method_handle(KernelLocallyLinearEmbedding);
        tapkee_method_handle(KernelLocalTangentSpaceAlignment);
        tapkee_method_handle(DiffusionMap);
        tapkee_method_handle(MultidimensionalScaling);
        tapkee_method_handle(LandmarkMultidimensionalScaling);
        tapkee_method_handle(Isomap);
        tapkee_method_handle(LandmarkIsomap);
        tapkee_method_handle(NeighborhoodPreservingEmbedding);
        tapkee_method_handle(LinearLocalTangentSpaceAlignment);
        tapkee_method_handle(HessianLocallyLinearEmbedding);
        tapkee_method_handle(LaplacianEigenmaps);
        tapkee_method_handle(LocalityPreservingProjections);
        tapkee_method_handle(PrincipalComponentAnalysis);
        tapkee_method_handle(KernelPrincipalComponentAnalysis);
        tapkee_method_handle(RandomProjection);
        tapkee_method_handle(StochasticProximityEmbedding);
        tapkee_method_handle(PassThru);
        tapkee_method_handle(FactorAnalysis);
        tapkee_method_handle(tDistributedStochasticNeighborEmbedding);
        tapkee_method_handle(ManifoldSculpting);
#undef tapkee_method_handle
        return TapkeeOutput();
    }
};

template <class RandomAccessIterator, class KernelCallback, class DistanceCallback, class FeaturesCallback>
DynamicImplementation<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback> initialize(
    RandomAccessIterator begin, RandomAccessIterator end, KernelCallback kernel, DistanceCallback distance,
    FeaturesCallback features, stichwort::ParametersSet& pmap, const Context& ctx)
{
    return DynamicImplementation<RandomAccessIterator, KernelCallback, DistanceCallback, FeaturesCallback>(
        begin, end, kernel, distance, features, pmap, ctx);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee
