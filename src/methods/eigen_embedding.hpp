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

#ifndef TAPKEE_EIGEN_EMBEDDING_H_
#define TAPKEE_EIGEN_EMBEDDING_H_

#include "../defines.hpp"
#include "../utils/time.hpp"
#include "../utils/arpack_wrapper.hpp"

/** Matrix-matrix operation used to 
 * compute smallest eigenvalues and 
 * associated eigenvectors. Essentially
 * solves linear system with provided
 * right-hand side part.
 */
template <class MatrixType>
struct InverseSparseMatrixOperation
{
	InverseSparseMatrixOperation(const MatrixType& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	inline DenseMatrix operator()(DenseMatrix operatee)
	{
		return solver.solve(operatee);
	}
	Eigen::SimplicialLDLT<MatrixType> solver;
};

/** Matrix-matrix operation used to
 * compute largest eigenvalues and
 * associated eigenvectors. Essentially
 * computes matrix product with 
 * provided right-hand side part.
 */
template <class MatrixType>
struct DenseMatrixOperation
{
	DenseMatrixOperation(const MatrixType& matrix) : _matrix(matrix)
	{
	}
	/** Computes matrix product of the matrix and provided right-hand 
	 * side matrix
	 */
	inline DenseMatrix operator()(DenseMatrix operatee)
	{
		return _matrix*operatee;
	}
	// TODO avoid copying somehow
	MatrixType _matrix;
};

/** Matrix-matrix operation used to
 * compute largest eigenvalues and
 * associated eigenvectors of X*X^T like
 * matrix implicitly. Essentially
 * computes matrix product with provided
 * right-hand side part *twice*.
 */
template <class MatrixType>
struct DenseImplicitSquareMatrixOperation
{
	DenseImplicitSquareMatrixOperation(const MatrixType& matrix) : _matrix(matrix)
	{
	}
	/** Computes matrix product of the matrix and provided right-hand 
	 * side matrix *twice*
	 */
	inline DenseMatrix operator()(DenseMatrix operatee)
	{
		return _matrix*(_matrix*operatee);
	}
	// TODO avoid copying somehow
	MatrixType _matrix;
};

/** Templated implementation of eigendecomposition-based embedding. 
 * Has three template parameters:
 * WeightMatrix - class of weight matrix to perform eigendecomposition of
 * WeightMatrixOperation - class of product operation over matrix.
 *
 * In order to find largest eigenvalues WeightMatrixOperation should provide
 * implementation of operator()(DenseMatrix) which computes right product
 * of the parameter with the WeightMatrix.
 */
template <class WeightMatrix, template<class> class WeightMatrixOperation, int> 
struct eigen_embedding_impl
{
	/** Construct embedding
	 * @param wm weight matrix to eigendecompose
	 * @param target_dimension target dimension of embedding (number of eigenvectors to find)
	 * @param skip number of eigenvectors to skip
	 */
	virtual EmbeddingResult embed(const WeightMatrix& wm, unsigned int target_dimension, unsigned int skip);
};

/** ARPACK implementation of eigendecomposition-based embedding */
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

/** Randomized redsvd-like implementation of eigendecomposition-based embedding */
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

/** Adapter method for various eigendecomposition methods. Currently
 * supports two methods:
 * * ARPACK_XSXUPD
 * * RANDOMIZED_INVERSE
 */
template <class WeightMatrix, template<class> class WeightMatrixOperation>
EmbeddingResult eigen_embedding(TAPKEE_EIGEN_EMBEDDING_METHOD method, const WeightMatrix& wm, 
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
