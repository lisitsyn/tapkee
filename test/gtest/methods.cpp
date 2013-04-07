#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>
#include <callback/eigen_callbacks.hpp>

#include "callbacks.hpp"
#include "data.hpp"

#include <vector>
#include <algorithm>
#include <set>

TEST(Methods,KernelLocallyLinearEmbeddingSmokeTest)
{
	const int N = 100;

	tapkee::DenseMatrix X = swissroll(N);

	kernel_callback kcb(X);
	distance_callback dcb(X);
	feature_vector_callback fcb(X);

	tapkee::ParametersMap parameters;
	parameters[tapkee::ReductionMethod] = tapkee::KernelLocallyLinearEmbedding;
	parameters[tapkee::CurrentDimension] = static_cast<tapkee::IndexType>(3);
	parameters[tapkee::TargetDimension] = static_cast<tapkee::IndexType>(2);
	parameters[tapkee::NumberOfNeighbors] = static_cast<tapkee::IndexType>(15);
	parameters[tapkee::NeighborsMethod] = tapkee::CoverTree;
	parameters[tapkee::EigenEmbeddingMethod] = tapkee::Arpack;
	//tapkee::LoggingSingleton::instance().enable_info();

	vector<int> data;
	for (int i=0; i<N; i++)
		data.push_back(i);

	tapkee::ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,parameters));
	// that's normal
	ASSERT_EQ(2,result.first.cols());
	// that's normal
	ASSERT_EQ(N,result.first.rows());
}

TEST(Methods,KernelLocalTangentSpaceAlignmentSmokeTest)
{
	const int N = 100;

	tapkee::DenseMatrix X = swissroll(N);

	kernel_callback kcb(X);
	distance_callback dcb(X);
	feature_vector_callback fcb(X);

	tapkee::ParametersMap parameters;
	parameters[tapkee::ReductionMethod] = tapkee::KernelLocalTangentSpaceAlignment;
	parameters[tapkee::CurrentDimension] = static_cast<tapkee::IndexType>(3);
	parameters[tapkee::TargetDimension] = static_cast<tapkee::IndexType>(2);
	parameters[tapkee::NumberOfNeighbors] = static_cast<tapkee::IndexType>(15);
	parameters[tapkee::NeighborsMethod] = tapkee::CoverTree;
	parameters[tapkee::EigenEmbeddingMethod] = tapkee::Arpack;
	//tapkee::LoggingSingleton::instance().enable_info();

	vector<int> data;
	for (int i=0; i<N; i++)
		data.push_back(i);

	tapkee::ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,parameters));
	// that's normal
	ASSERT_EQ(2,result.first.cols());
	// that's normal
	ASSERT_EQ(N,result.first.rows());
}

TEST(Methods,HessianLocallyLinearEmbeddingSmokeTest)
{
	const int N = 100;

	tapkee::DenseMatrix X = swissroll(N);

	kernel_callback kcb(X);
	distance_callback dcb(X);
	feature_vector_callback fcb(X);

	tapkee::ParametersMap parameters;
	parameters[tapkee::ReductionMethod] = tapkee::HessianLocallyLinearEmbedding;
	parameters[tapkee::CurrentDimension] = static_cast<tapkee::IndexType>(3);
	parameters[tapkee::TargetDimension] = static_cast<tapkee::IndexType>(2);
	parameters[tapkee::NumberOfNeighbors] = static_cast<tapkee::IndexType>(15);
	parameters[tapkee::NeighborsMethod] = tapkee::CoverTree;
	parameters[tapkee::EigenEmbeddingMethod] = tapkee::Arpack;
	//tapkee::LoggingSingleton::instance().enable_info();

	vector<int> data;
	for (int i=0; i<N; i++)
		data.push_back(i);

	tapkee::ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,parameters));
	// that's normal
	ASSERT_EQ(2,result.first.cols());
	// that's normal
	ASSERT_EQ(N,result.first.rows());
}


