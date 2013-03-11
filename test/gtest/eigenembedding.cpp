#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>

#include "callbacks.hpp"

#include <cmath>

#include <Eigen/Dense>

#define PRECISION 1e-7

TEST(EigenEmbedding, EigenDenseLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);

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

TEST(EigenEmbedding, EigenSparseSmallestEigenvector) 
{
	tapkee::SparseWeightMatrix mat(3,3);
	tapkee::tapkee_internal::SparseTriplets triplets;
	for (int i=0; i<3; i++)
		triplets.push_back(tapkee::tapkee_internal::SparseTriplet(i,i,tapkee::ScalarType(i+1)));
	mat.setFromTriplets(triplets.begin(), triplets.end());

	tapkee::tapkee_internal::EmbeddingResult result = 
		tapkee::tapkee_internal::eigen_embedding<tapkee::SparseWeightMatrix,tapkee::tapkee_internal::SparseInverseMatrixOperation>
		(tapkee::EIGEN_DENSE_SELFADJOINT_SOLVER, mat, 1, 0);

	ASSERT_EQ(1,result.second.size());
	// smallest eigenvalue is 1
	ASSERT_NEAR(1,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(3,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenEmbedding, ArpackDenseLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);
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

TEST(EigenEmbedding, ArpackSparseSmallestEigenvector) 
{
	tapkee::SparseWeightMatrix mat(3,3);
	tapkee::tapkee_internal::SparseTriplets triplets;
	for (int i=0; i<3; i++)
		triplets.push_back(tapkee::tapkee_internal::SparseTriplet(i,i,tapkee::ScalarType(i+1)));
	mat.setFromTriplets(triplets.begin(), triplets.end());

	tapkee::tapkee_internal::EmbeddingResult result = 
		tapkee::tapkee_internal::eigen_embedding<tapkee::SparseWeightMatrix,tapkee::tapkee_internal::SparseInverseMatrixOperation>
		(tapkee::ARPACK, mat, 1, 0);

	ASSERT_EQ(1,result.second.size());
	// smallest eigenvalue is 1
	ASSERT_NEAR(1,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(3,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenEmbedding, RandomizedDenseLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);
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
