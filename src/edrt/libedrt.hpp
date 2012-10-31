#ifndef libedrt_h_
#define libedrt_h_

#define HAVE_LAPACK
#define HAVE_ARPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/lib/Time.h>
#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>

#include <vector>
#include <algorithm>
#include <iostream>

#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SuperLUSupport>
#include "libedrt_defines.hpp"
#include "libedrt_methods.hpp"
#include "libedrt_neighbors.hpp"
#include "libedrt_embedding.hpp"

using std::cout;
using std::endl;


enum edrt_method_t
{
	KERNEL_LOCALLY_LINEAR_EMBEDDING,
	NEIGHBORHOOD_PRESERVING_EMBEDDING,
	KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT,
	LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT,
	HESSIAN_LOCALLY_LINEAR_EMBEDDING,
	LAPLACIAN_EIGENMAPS,
	LOCALITY_PRESERVING_PROJECTIONS,
	DIFFUSION_MAPS,
	ISOMAP,
	MULTIDIMENSIONAL_SCALING,
	STOCHASTIC_PROXIMITY_EMBEDDING,
	MAXIMUM_VARIANCE_UNFOLDING
};

struct edrt_options_t
{
	edrt_options_t()
	{
		method = KERNEL_LOCALLY_LINEAR_EMBEDDING;
		num_threads = 1;
		use_arpack = true;
		use_superlu = true;
		mds_use_landmarks = false;
		klle_reconstruction_shift = 1e-3;
		diffusion_maps_t = 1;
		nullspace_shift = 1e-9;
	}
	// EDRT method
	edrt_method_t method;
	// number of threads
	int num_threads;
	// true if ARPACK should be used
	bool use_arpack;
	// true if SuperLU should be used
	bool use_superlu;
	// mds use landmarks
	bool mds_use_landmarks;
	// kernel LLE reconstruction shift
	double klle_reconstruction_shift;
	// diffusion maps t
	int diffusion_maps_t;
	// nullspace regularization shift
	double nullspace_shift;
};

EmbeddingMatrix eigen_embedding(WeightMatrix wm, unsigned int target_dimension)
{
	/*
	/// ARPACK based eigendecomposition
	int N = wm.cols();
	double* eigenvalues_vector = SG_MALLOC(double, N);
	double* eigenvectors = SG_MALLOC(double, (target_dimension+1)*N);
	int eigenproblem_status = 0;
	Eigen::MatrixXd weight_matrix = Eigen::MatrixXd::Zero(N,N);
	weight_matrix += wm;
	shogun::arpack_dsxupd(weight_matrix.data(), NULL, false, N, target_dimension+1,
	                      "LA", true, 3, true, false, -1e-3, 0.0,
	                      eigenvalues_vector, weight_matrix.data(), eigenproblem_status);
	Eigen::MatrixXd embedding_feature_matrix = Eigen::MatrixXd::Zero(N,target_dimension);
	for (int i=0; i<target_dimension; i++)
	{
		for (int j=0; j<N; j++)
			embedding_feature_matrix(j,i) = 
				weight_matrix.data()[j*(target_dimension+1)+i+1];
	}
	SG_FREE(eigenvalues_vector);
	SG_FREE(eigenvectors);
	return embedding_feature_matrix;
	*/
	/*
	/// LAPACK based eigendecomposition
	int N = wm.cols();
	double* eigenvalues_vector = SG_MALLOC(double, N);
	double* eigenvectors = SG_MALLOC(double, (target_dimension+1)*N);
	int eigenproblem_status = 0;
	Eigen::MatrixXd weight_matrix = Eigen::MatrixXd::Zero(N,N);
	weight_matrix += wm;
	weight_matrix.diagonal().array() -= 1e-5;
	shogun::wrap_dsyevr('V','U',N,weight_matrix.data(),N,2,target_dimension+2,
	                    eigenvalues_vector,eigenvectors,&eigenproblem_status);
	EmbeddingMatrix embedding_feature_matrix(target_dimension,N);
	for (int i=0; i<target_dimension; i++)
	{
		for (int j=0; j<N; j++)
			embedding_feature_matrix(i,j) = eigenvectors[i*N+j];
	}
	SG_FREE(eigenvectors);
	SG_FREE(eigenvalues_vector);
	return embedding_feature_matrix.transpose();
	*/

	cout << "Embedding" << endl;
	shogun::CTime time(true);
	Eigen::MatrixXd O(wm.rows(), target_dimension+1);
	for (int i=0; i<O.rows(); ++i)
	{
		int j=0;
		for ( ; j+1 < O.cols(); j+= 2)
		{
			double v1 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double v2 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double len = sqrt(-2.f*log(v1));
			O(i,j) = len*cos(2.f*M_PI*v2);
			O(i,j+1) = len*sin(2.f*M_PI*v2);
		}
		for ( ; j < O.cols(); j++)
		{
			double v1 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double v2 = (double)(rand()+1.f)/((float)RAND_MAX+2.f);
			double len = sqrt(-2.f*log(v1));
			O(i,j) = len*cos(2.f*M_PI*v2);
		}
	}
	cout << "Solver creation" << endl;
	Eigen::SuperLU<WeightMatrix> solver;
	solver.compute(wm);
	cout << "Factorized" << endl;

	Eigen::MatrixXd Y = solver.solve(O);
	cout << "Solved" << endl;
	for (int i=0; i<Y.cols(); i++)
	{
		for (int j=0; j<i; j++)
		{
			double r = Y.col(i).dot(Y.col(j));
			Y.col(i) -= r*Y.col(j);
		}
		double norm = Y.col(i).norm();
		if (norm < 1e-4)
		{
			for (int k = i; k<Y.cols(); k++)
				Y.col(k).setZero();
		}
		Y.col(i) *= (1.f / norm);
	}

	DenseMatrix B1 = solver.solve(Y);
	DenseMatrix B = Y.householderQr().solve(B1);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenOfB(B);
	Eigen::MatrixXd embedding = Y*eigenOfB.eigenvectors();
	cout << "eigenproblem took " << time.cur_time_diff() << endl;
	cout << "embedding # cols " << embedding.cols() << endl;
	cout << "embedding # rows " << embedding.rows() << endl;
	return embedding.block(0, 1, wm.cols(), target_dimension);

}

template <class RandomAccessIterator, class PairwiseCallback>
Eigen::MatrixXd embed(
		RandomAccessIterator begin,
		RandomAccessIterator end,
		PairwiseCallback callback,
		const edrt_options_t& options,
		const int target_dimension,
		const int dimension,
		const int k)
{
	Neighbors neighbors;
	WeightMatrix weight_matrix;
	EmbeddingMatrix embedding_matrix;

	switch (options.method)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING:
			neighbors = find_neighbors(begin,end,callback,k);
			weight_matrix = klle_weight_matrix(begin,end,neighbors,callback);
			embedding_matrix = eigen_embedding(weight_matrix,target_dimension);
			break;
		case NEIGHBORHOOD_PRESERVING_EMBEDDING:
			break;
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT:
			neighbors = find_neighbors(begin,end,callback,k);
			weight_matrix = kltsa_weight_matrix(begin,end,neighbors,callback,target_dimension);
			embedding_matrix = eigen_embedding(weight_matrix,target_dimension);
			break;
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT:
			break;
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING:
			break;
		case LAPLACIAN_EIGENMAPS:
			break;
		case LOCALITY_PRESERVING_PROJECTIONS:
			break;
		case DIFFUSION_MAPS:
			break;
		case ISOMAP:
			break;
		case MULTIDIMENSIONAL_SCALING:
			break;
		default:
			break;
	}
	return embedding_matrix;
}

#endif /* libedrt_h_ */
