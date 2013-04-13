#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>
#include <callback/eigen_callbacks.hpp>

#include "callbacks.hpp"
#include "data.hpp"

#include <vector>
#include <algorithm>
#include <set>

using namespace tapkee;
using namespace tapkee::keywords;
using std::vector;

TEST(Methods,KernelLocallyLinearEmbeddingSmokeTest)
{
	const int N = 100;

	DenseMatrix X = swissroll(N);

	kernel_callback kcb(X);
	distance_callback dcb(X);
	feature_vector_callback fcb(X);

	vector<int> data;
	for (int i=0; i<N; i++)
		data.push_back(i);

	ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,
	                               (method=KernelLocallyLinearEmbedding,num_neighbors=15)));
	// that's normal
	ASSERT_EQ(2,result.first.cols());
	// that's normal
	ASSERT_EQ(N,result.first.rows());
}

TEST(Methods,KernelLocalTangentSpaceAlignmentSmokeTest)
{
	const int N = 100;

	DenseMatrix X = swissroll(N);

	kernel_callback kcb(X);
	distance_callback dcb(X);
	feature_vector_callback fcb(X);

	vector<int> data;
	for (int i=0; i<N; i++)
		data.push_back(i);

	tapkee::ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,
	                               (method=KernelLocalTangentSpaceAlignment,num_neighbors=15)));
	// that's normal
	ASSERT_EQ(2,result.first.cols());
	// that's normal
	ASSERT_EQ(N,result.first.rows());
}

TEST(Methods,HessianLocallyLinearEmbeddingSmokeTest)
{
	const int N = 100;

	DenseMatrix X = swissroll(N);

	kernel_callback kcb(X);
	distance_callback dcb(X);
	feature_vector_callback fcb(X);

	vector<int> data;
	for (int i=0; i<N; i++)
		data.push_back(i);

	tapkee::ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,
	                               (method=HessianLocallyLinearEmbedding,num_neighbors=15)));
	// that's normal
	ASSERT_EQ(2,result.first.cols());
	// that's normal
	ASSERT_EQ(N,result.first.rows());
}


