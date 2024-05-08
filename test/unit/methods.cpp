#include <gtest/gtest.h>

#include <tapkee/callbacks/eigen_callbacks.hpp>
#include <tapkee/exceptions.hpp>
#include <tapkee/tapkee.hpp>

#include "callbacks.hpp"
#include "data.hpp"

#include <algorithm>
#include <set>
#include <vector>

using namespace tapkee;

void smoketest(DimensionReductionMethod m)
{
    const int N = 100;
    DenseMatrix X = swissroll(N);
    tapkee::eigen_kernel_callback kcb(X);
    tapkee::eigen_distance_callback dcb(X);
    tapkee::eigen_features_callback fcb(X);
    std::vector<int> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = i;
    TapkeeOutput result;
    ASSERT_NO_THROW(result = embed(data.begin(), data.end(), kcb, dcb, fcb,
                                   (method = m, target_dimension = 2, num_neighbors = N / 5,
                                    gaussian_kernel_width = 10.0, sne_perplexity = 10.0)));
    ASSERT_EQ(2, result.embedding.cols());
    ASSERT_EQ(N, result.embedding.rows());
}

TEST(Methods, KernelLocallyLinearEmbeddingSmokeTest)
{
    smoketest(KernelLocallyLinearEmbedding);
}

TEST(Methods, NeighborhoodPreservingEmbedding)
{
    smoketest(NeighborhoodPreservingEmbedding);
}

TEST(Methods, KernelLocalTangentSpaceAlignmentSmokeTest)
{
    smoketest(KernelLocallyLinearEmbedding);
}

TEST(Methods, LinearLocalTangentSpaceAlignment)
{
    smoketest(LinearLocalTangentSpaceAlignment);
}

TEST(Methods, HessianLocallyLinearEmbeddingSmokeTest)
{
    smoketest(HessianLocallyLinearEmbedding);
}

TEST(Methods, LaplacianEigenmapsSmokeTest)
{
    smoketest(LaplacianEigenmaps);
}

TEST(Methods, LocalityPreservingProjectionsSmokeTest)
{
    smoketest(LocalityPreservingProjections);
}

TEST(Methods, DiffusionMapSmokeTest)
{
    smoketest(DiffusionMap);
}

TEST(Methods, IsomapSmokeTest)
{
    smoketest(Isomap);
}

TEST(Methods, LandmarkIsomapSmokeTest)
{
    smoketest(LandmarkIsomap);
}

TEST(Methods, MultidimensionalScalingSmokeTest)
{
    smoketest(MultidimensionalScaling);
}

TEST(Methods, LandmarkMultidimensionalScalingSmokeTest)
{
    smoketest(LandmarkMultidimensionalScaling);
}

TEST(Methods, StochasticProximityEmbeddingSmokeTest)
{
    smoketest(StochasticProximityEmbedding);
}

TEST(Methods, KernelPrincipalComponentAnalysisSmokeTest)
{
    smoketest(KernelPrincipalComponentAnalysis);
}

TEST(Methods, PrincipalComponentAnalysisSmokeTest)
{
    smoketest(PrincipalComponentAnalysis);
}

TEST(Methods, RandomProjectionSmokeTest)
{
    smoketest(RandomProjection);
}

TEST(Methods, FactorAnalysisSmokeTest)
{
    smoketest(FactorAnalysis);
}

TEST(Methods, tDistributedStochasticNeighborEmbeddingSmokeTest)
{
    smoketest(tDistributedStochasticNeighborEmbedding);
}
