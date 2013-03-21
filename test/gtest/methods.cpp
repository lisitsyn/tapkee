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
	parameters[tapkee::REDUCTION_METHOD] = tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING;
	parameters[tapkee::CURRENT_DIMENSION] = static_cast<tapkee::IndexType>(3);
	parameters[tapkee::TARGET_DIMENSION] = static_cast<tapkee::IndexType>(2);
	parameters[tapkee::NUMBER_OF_NEIGHBORS] = static_cast<tapkee::IndexType>(15);
	parameters[tapkee::NEIGHBORS_METHOD] = tapkee::COVER_TREE;
	parameters[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee::ARPACK;
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
	parameters[tapkee::REDUCTION_METHOD] = tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
	parameters[tapkee::CURRENT_DIMENSION] = static_cast<tapkee::IndexType>(3);
	parameters[tapkee::TARGET_DIMENSION] = static_cast<tapkee::IndexType>(2);
	parameters[tapkee::NUMBER_OF_NEIGHBORS] = static_cast<tapkee::IndexType>(15);
	parameters[tapkee::NEIGHBORS_METHOD] = tapkee::COVER_TREE;
	parameters[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee::ARPACK;
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
	parameters[tapkee::REDUCTION_METHOD] = tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING;
	parameters[tapkee::CURRENT_DIMENSION] = static_cast<tapkee::IndexType>(3);
	parameters[tapkee::TARGET_DIMENSION] = static_cast<tapkee::IndexType>(2);
	parameters[tapkee::NUMBER_OF_NEIGHBORS] = static_cast<tapkee::IndexType>(15);
	parameters[tapkee::NEIGHBORS_METHOD] = tapkee::COVER_TREE;
	parameters[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee::ARPACK;
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


