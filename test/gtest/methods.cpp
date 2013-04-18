#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>
#include <callbacks/eigen_callbacks.hpp>

#include "callbacks.hpp"
#include "data.hpp"

#include <vector>
#include <algorithm>
#include <set>

using namespace tapkee;
using namespace tapkee::keywords;
using std::vector;

void smoketest(DimensionReductionMethod m)
{
	const int N = 50;
	DenseMatrix X = swissroll(N);
	tapkee::eigen_kernel_callback kcb(X);
	tapkee::eigen_distance_callback dcb(X);
	tapkee::eigen_features_callback fcb(X);
	vector<int> data(N);
	for (int i=0; i<N; ++i) data[i] = i;
	ReturnResult result;
	ASSERT_NO_THROW(result = embed(data.begin(), data.end(),
		kcb, dcb, fcb, (method=m,target_dimension=2,num_neighbors=N/5,
	                    gaussian_kernel_width=10.0,sne_perplexity=10.0)));
	ASSERT_EQ(2,result.first.cols());
	ASSERT_EQ(N,result.first.rows());
}

TEST(Methods,KernelLocallyLinearEmbeddingSmokeTest)
{
	smoketest(KernelLocallyLinearEmbedding);
}

TEST(Methods,NeighborhoodPreservingEmbedding)
{
	smoketest(NeighborhoodPreservingEmbedding);
}

TEST(Methods,KernelLocalTangentSpaceAlignmentSmokeTest)
{
	smoketest(KernelLocallyLinearEmbedding);
}

TEST(Methods,LinearLocalTangentSpaceAlignment)
{
	smoketest(LinearLocalTangentSpaceAlignment);
}

TEST(Methods,HessianLocallyLinearEmbeddingSmokeTest)
{
	smoketest(HessianLocallyLinearEmbedding);
}

TEST(Methods,LaplacianEigenmapsSmokeTest)
{
	smoketest(LaplacianEigenmaps);
}

TEST(Methods,LocalityPreservingProjectionsSmokeTest)
{
	smoketest(LocalityPreservingProjections);
}

TEST(Methods,DiffusionMapSmokeTest)
{
	smoketest(DiffusionMap);
}

TEST(Methods,IsomapSmokeTest)
{
	smoketest(Isomap);
}

TEST(Methods,LandmarkIsomapSmokeTest)
{
	smoketest(LandmarkIsomap);
}

TEST(Methods,MultidimensionalScalingSmokeTest)
{
	smoketest(MultidimensionalScaling);
}

TEST(Methods,LandmarkMultidimensionalScalingSmokeTest)
{
	smoketest(LandmarkMultidimensionalScaling);
}
		
TEST(Methods,StochasticProximityEmbeddingSmokeTest)
{
	smoketest(StochasticProximityEmbedding);
}

TEST(Methods,KernelPCASmokeTest)
{
	smoketest(KernelPCA);
}

TEST(Methods,PCASmokeTest)
{
	smoketest(PCA);
}

TEST(Methods,RandomProjectionSmokeTest)
{
	smoketest(RandomProjection);
}

TEST(Methods,FactorAnalysisSmokeTest)
{
	smoketest(FactorAnalysis);
}

TEST(Methods,tDistributedStochasticNeighborEmbeddingSmokeTest)
{
	smoketest(tDistributedStochasticNeighborEmbedding);
}
