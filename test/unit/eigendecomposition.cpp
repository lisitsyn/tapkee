#include <gtest/gtest.h>

#include <tapkee/tapkee.hpp>
#include <tapkee/exceptions.hpp>

#include "callbacks.hpp"

#include <cmath>

#define PRECISION 1e-7

TEST(EigenDecomposition, EigenDenseLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);

	mat << 1, -1, -1, 1;

	tapkee::tapkee_internal::EigendecompositionResult result = 
		tapkee::tapkee_internal::eigendecomposition
		(tapkee::Dense, tapkee::HomogeneousCPUStrategy, tapkee::tapkee_internal::LargestEigenvalues, mat, 1);

	ASSERT_EQ(1,result.second.size());
	// eigenvalue is 2
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenDecomposition, EigenSparseSmallestEigenvector) 
{
	const int N = 3;
	tapkee::tapkee_internal::SparseTriplets sparse_triplets;
	for (int i=0; i<N; i++)
		sparse_triplets.push_back(tapkee::tapkee_internal::SparseTriplet(i,i,tapkee::ScalarType(i+1)));

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	Eigen::DynamicSparseMatrix<tapkee::ScalarType> dynamic_weight_matrix(N,N);
	dynamic_weight_matrix.reserve(sparse_triplets.size());
	for (tapkee::tapkee_internal::SparseTriplets::const_iterator it=sparse_triplets.begin(); it!=sparse_triplets.end(); ++it)
		dynamic_weight_matrix.coeffRef(it->col(),it->row()) += it->value();
	tapkee::SparseWeightMatrix mat(dynamic_weight_matrix);
#else
	tapkee::SparseWeightMatrix mat(N,N);
	mat.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());
#endif

	tapkee::tapkee_internal::EigendecompositionResult result = 
		tapkee::tapkee_internal::eigendecomposition
		(tapkee::Dense, tapkee::HomogeneousCPUStrategy, tapkee::tapkee_internal::SmallestEigenvalues, mat, 1);

	ASSERT_EQ(2,result.second.size());
	// smallest eigenvalue is 1
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(3,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenDecomposition, ArpackDenseLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);
	mat << 1, -1, -1, 1;

	tapkee::tapkee_internal::EigendecompositionResult result = 
		tapkee::tapkee_internal::eigendecomposition
		(tapkee::Arpack, tapkee::HomogeneousCPUStrategy, tapkee::tapkee_internal::LargestEigenvalues, mat, 1);

	ASSERT_EQ(1,result.second.size());
	// eigenvalue is 2
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenDecomposition, ArpackSparseSmallestEigenvector) 
{
	const int N = 3;
	tapkee::tapkee_internal::SparseTriplets sparse_triplets;
	for (int i=0; i<N; i++)
		sparse_triplets.push_back(tapkee::tapkee_internal::SparseTriplet(i,i,tapkee::ScalarType(i+1)));

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	Eigen::DynamicSparseMatrix<tapkee::ScalarType> dynamic_weight_matrix(N,N);
	dynamic_weight_matrix.reserve(sparse_triplets.size());
	for (tapkee::tapkee_internal::SparseTriplets::const_iterator it=sparse_triplets.begin(); it!=sparse_triplets.end(); ++it)
		dynamic_weight_matrix.coeffRef(it->col(),it->row()) += it->value();
	tapkee::SparseWeightMatrix mat(dynamic_weight_matrix);
#else
	tapkee::SparseWeightMatrix mat(N,N);
	mat.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());
#endif

	tapkee::tapkee_internal::EigendecompositionResult result = 
		tapkee::tapkee_internal::eigendecomposition
		(tapkee::Arpack, tapkee::HomogeneousCPUStrategy, tapkee::tapkee_internal::SmallestEigenvalues, mat, 1);

	ASSERT_EQ(1,result.second.size());
	// smallest eigenvalue is 1
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(3,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}

TEST(EigenDecomposition, RandomizedDenseLargestEigenvector) 
{
	tapkee::DenseMatrix mat(2,2);
	mat << 1, -1, -1, 1;

	tapkee::tapkee_internal::EigendecompositionResult result = 
		tapkee::tapkee_internal::eigendecomposition
		(tapkee::Randomized, tapkee::HomogeneousCPUStrategy, tapkee::tapkee_internal::LargestEigenvalues, mat, 1);

	ASSERT_EQ(1,result.second.size());
	// eigenvalue is 2
	ASSERT_NEAR(2,result.second[0],PRECISION);
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
	// check if it is an eigenvector
	ASSERT_NEAR(0.0,(mat*result.first - result.second[0]*result.first).norm(),PRECISION);
}
