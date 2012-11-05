/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 *
 * Randomized eigendecomposition code is inspired by the redsvd library
 * code which is distributed under BSD 3-clause license.
 *
 * Copyright (c) 2010, Daisuke Okanohara
 *
 */

#ifndef EDRT_EIGEN_EMBEDDING_H_
#define EDRT_EIGEN_EMBEDDING_H_

#include "../defines.hpp"
#include "../utils/time.hpp"
#include "../utils/arpack_wrapper.hpp"

template <class MatrixType>
struct InverseSparseMatrixOperation
{
	InverseSparseMatrixOperation(const MatrixType& matrix) : solver()
	{
		solver.compute(matrix);
	}
	DenseMatrix operator()(DenseMatrix operatee)
	{
		return solver.solve(operatee);
	}
	Eigen::SimplicialLDLT<MatrixType> solver;
};

template <class MatrixType>
struct DenseMatrixOperation
{
	DenseMatrixOperation(const MatrixType& matrix) : _matrix(matrix)
	{
	}
	DenseMatrix operator()(DenseMatrix operatee)
	{
		return _matrix*operatee;
	}
	MatrixType _matrix;
};

template <class WeightMatrix, template<class> class WeightMatrixOperation, int> 
struct eigen_embedding_impl
{
	virtual EmbeddingResult embed(const WeightMatrix& wm, unsigned int target_dimension, unsigned int skip);
};

template <class WeightMatrix, template<class> class WeightMatrixOperation> 
struct eigen_embedding_impl<WeightMatrix, WeightMatrixOperation, ARPACK_XSXUPD>
{
	EmbeddingResult embed(const WeightMatrix& wm, unsigned int target_dimension, unsigned int skip)
	{
		timed_context context("ARPACK DSXUPD eigendecomposition");

		ArpackGeneralizedSelfAdjointEigenSolver<WeightMatrix, WeightMatrixOperation> arpack(wm,target_dimension+skip,"SM");

		DenseMatrix embedding_feature_matrix = (arpack.eigenvectors()).block(0,skip,wm.cols(),target_dimension);

		return EmbeddingResult(embedding_feature_matrix,arpack.eigenvalues().tail(target_dimension));
	}
};

template <class WeightMatrix, template<class> class WeightMatrixOperation> 
struct eigen_embedding_impl<WeightMatrix, WeightMatrixOperation, RANDOMIZED_INVERSE>
{
	EmbeddingResult embed(const WeightMatrix& wm, unsigned int target_dimension, unsigned int skip)
	{
		timed_context context("Randomized eigendecomposition");
		
		DenseMatrix O(wm.rows(), target_dimension+skip);
		for (unsigned int i=0; i<O.rows(); ++i)
		{
			unsigned int j=0;
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
		WeightMatrixOperation<WeightMatrix> operation(wm);

		DenseMatrix Y = operation(O);
		for (unsigned int i=0; i<Y.cols(); i++)
		{
			for (unsigned int j=0; j<i; j++)
			{
				DefaultScalarType r = Y.col(i).dot(Y.col(j));
				Y.col(i) -= r*Y.col(j);
			}
			DefaultScalarType norm = Y.col(i).norm();
			if (norm < 1e-4)
			{
				for (int k = i; k<Y.cols(); k++)
					Y.col(k).setZero();
			}
			Y.col(i) *= (1.f / norm);
		}

		DenseMatrix B1 = operation(Y);
		DenseMatrix B = Y.householderQr().solve(B1);
		Eigen::SelfAdjointEigenSolver<DenseMatrix> eigenOfB(B);
		DenseMatrix embedding = (Y*eigenOfB.eigenvectors()).block(0, skip, wm.cols(), target_dimension);

		/*
		DenseMatrix covariance(target_dimension,target_dimension);
		covariance = embedding.transpose()*embedding;
		//covariance.centerMatrix();
		Eigen::SelfAdjointEigenSolver<DenseMatrix> pca(covariance);
		embedding *= pca.eigenvectors();
		*/
		/* refinements idea (drop probably)
		 */
		return EmbeddingResult(embedding,eigenOfB.eigenvalues());
	}
};

template <class WeightMatrix, template<class> class WeightMatrixOperation>
EmbeddingResult eigen_embedding(EDRT_EIGEN_EMBEDDING_METHOD method, const WeightMatrix& wm, 
                                unsigned int target_dimension, unsigned int skip)
{
	switch (method)
	{
		case ARPACK_XSXUPD: 
			return eigen_embedding_impl<WeightMatrix, WeightMatrixOperation, 
				ARPACK_XSXUPD>().embed(wm, target_dimension, skip);
		case RANDOMIZED_INVERSE: 
			return eigen_embedding_impl<WeightMatrix, WeightMatrixOperation,
				RANDOMIZED_INVERSE>().embed(wm, target_dimension, skip);
	}
	return EmbeddingResult();
};
#endif
