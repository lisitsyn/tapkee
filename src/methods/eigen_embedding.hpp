/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef EDRT_EIGEN_EMBEDDING_H_
#define EDRT_EIGEN_EMBEDDING_H_

#include "../defines.hpp"
#include "../utils/time.hpp"

#include <eigen3/Eigen/SuperLUSupport>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>

template <int> struct eigen_embedding_impl
{
	virtual EmbeddingMatrix embed(const WeightMatrix& wm, unsigned int target_dimension);
};

template <> struct eigen_embedding_impl<ARPACK_XSXUPD>
{
	EmbeddingMatrix embed(const WeightMatrix& wm, unsigned int target_dimension)
	{
		timed_context context("ARPACK DSXUPD eigendecomposition");
		unsigned int N = wm.cols();
		double* eigenvalues_vector = SG_MALLOC(double, N);
		double* eigenvectors = SG_MALLOC(double, (target_dimension+1)*N);
		int eigenproblem_status = 0;
		DenseMatrix weight_matrix = DenseMatrix::Zero(N,N);
		weight_matrix += wm;
		shogun::arpack_dsxupd(weight_matrix.data(), NULL, false, N, target_dimension+1,
							  "LA", true, 3, true, false, 0.0, 0.0,
							  eigenvalues_vector, weight_matrix.data(), eigenproblem_status);
		DenseMatrix embedding_feature_matrix = DenseMatrix::Zero(N,target_dimension);
		for (unsigned int i=0; i<target_dimension; i++)
		{
			for (unsigned int j=0; j<N; j++)
				embedding_feature_matrix(j,i) = 
					weight_matrix.data()[j*(target_dimension+1)+i+1];
		}
		SG_FREE(eigenvalues_vector);
		SG_FREE(eigenvectors);
		return embedding_feature_matrix;
	}
};

template <> struct eigen_embedding_impl<LAPACK_XSYEVR>
{
	EmbeddingMatrix embed(const WeightMatrix& wm, unsigned int target_dimension)
	{
		timed_context context("LAPACK DSYEVR eigendecomposition");
		unsigned int N = wm.cols();
		double* eigenvalues_vector = SG_MALLOC(double, N);
		double* eigenvectors = SG_MALLOC(double, (target_dimension+1)*N);
		int eigenproblem_status = 0;
		Eigen::MatrixXd weight_matrix = Eigen::MatrixXd::Zero(N,N);
		weight_matrix += wm;
		weight_matrix.diagonal().array() -= 1e-5;
		shogun::wrap_dsyevr('V','U',N,weight_matrix.data(),N,2,target_dimension+2,
							eigenvalues_vector,eigenvectors,&eigenproblem_status);
		EmbeddingMatrix embedding_feature_matrix(target_dimension,N);
		for (unsigned int i=0; i<target_dimension; i++)
		{
			for (unsigned int j=0; j<N; j++)
				embedding_feature_matrix(i,j) = eigenvectors[i*N+j];
		}
		SG_FREE(eigenvectors);
		SG_FREE(eigenvalues_vector);
		return embedding_feature_matrix.transpose();
	}
};

template <> struct eigen_embedding_impl<RANDOMIZED_INVERSE>
{
	EmbeddingMatrix embed(const WeightMatrix& wm, unsigned int target_dimension)
	{
		const int eigenvalues_skip = 1;

		timed_context context("Randomized eigendecomposition");
		
		DenseMatrix O(wm.rows(), target_dimension+eigenvalues_skip);
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
		// TODO parametrize or define
		Eigen::SuperLU<WeightMatrix> solver;
		solver.compute(wm);

		DenseMatrix Y = solver.solve(O);
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
		DenseMatrix embedding = Y*eigenOfB.eigenvectors();

		/* refinements idea (drop probably)
		const int n_refinements = 20;
		for (int r=0; r<n_refinements; r++)
		{
			embedding = solver.solve(embedding);
			embedding /= embedding.norm();
		}
		*/
		return embedding.block(0, eigenvalues_skip, wm.cols(), target_dimension);
	}
};

EmbeddingMatrix eigen_embedding(EDRT_EIGEN_EMBEDDING_METHOD method, const WeightMatrix& wm, unsigned int target_dimension)
{
	switch (method)
	{
		case ARPACK_XSXUPD: return eigen_embedding_impl<ARPACK_XSXUPD>().embed(wm, target_dimension);
		case LAPACK_XSYEVR: return eigen_embedding_impl<LAPACK_XSYEVR>().embed(wm, target_dimension);
		case RANDOMIZED_INVERSE: return eigen_embedding_impl<RANDOMIZED_INVERSE>().embed(wm, target_dimension);
	}
	return EmbeddingMatrix();
};
#endif
