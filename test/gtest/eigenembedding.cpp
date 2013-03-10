#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>

#include "callbacks.hpp"

#include <cmath>

#include <Eigen/Dense>

#define PRECISION 1e-7

TEST(EigenEmbedding, EigenLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);

	// eigenv
	mat << 1, -1, -1, 1;

	tapkee::tapkee_internal::EmbeddingResult result = 
		tapkee::tapkee_internal::eigen_embedding<tapkee::DenseMatrix,tapkee::tapkee_internal::DenseMatrixOperation>
		(tapkee::EIGEN_DENSE_SELFADJOINT_SOLVER, mat, 1, 0);

	ASSERT_EQ(1,result.second.size());
	// eigenvalue is 2
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenEmbedding, ArpackLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);

	// eigenv
	mat << 1, -1, -1, 1;

	tapkee::tapkee_internal::EmbeddingResult result = 
		tapkee::tapkee_internal::eigen_embedding<tapkee::DenseMatrix,tapkee::tapkee_internal::DenseMatrixOperation>
		(tapkee::ARPACK, mat, 1, 0);

	ASSERT_EQ(1,result.second.size());
	// eigenvalue is 2
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenEmbedding, RandomizedLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);

	// eigenv
	mat << 1, -1, -1, 1;

	tapkee::tapkee_internal::EmbeddingResult result = 
		tapkee::tapkee_internal::eigen_embedding<tapkee::DenseMatrix,tapkee::tapkee_internal::DenseMatrixOperation>
		(tapkee::RANDOMIZED, mat, 1, 0);

	ASSERT_EQ(1,result.second.size());
	// eigenvalue is 2
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}
