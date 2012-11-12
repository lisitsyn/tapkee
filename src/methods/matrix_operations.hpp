/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_MATRIX_OPS_H_
#define TAPKEE_MATRIX_OPS_H_

#include "../defines.hpp"

/** Matrix-matrix operation used to 
 * compute smallest eigenvalues and 
 * associated eigenvectors. Essentially
 * solves linear system with provided
 * right-hand side part.
 */
struct InverseSparseMatrixOperation
{
	InverseSparseMatrixOperation(const SparseWeightMatrix& matrix) : solver()
	{
		solver.compute(matrix);
	}
	/** Solves linear system with provided right-hand size
	 */
	inline DenseMatrix operator()(DenseMatrix operatee)
	{
		return solver.solve(operatee);
	}
	DefaultSparseSolver solver;
};

/** Matrix-matrix operation used to
 * compute largest eigenvalues and
 * associated eigenvectors. Essentially
 * computes matrix product with 
 * provided right-hand side part.
 */
struct DenseMatrixOperation
{
	DenseMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	/** Computes matrix product of the matrix and provided right-hand 
	 * side matrix
	 */
	inline DenseMatrix operator()(DenseMatrix operatee)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*operatee;
	}
	const DenseMatrix& _matrix;
};

/** Matrix-matrix operation used to
 * compute largest eigenvalues and
 * associated eigenvectors of X*X^T like
 * matrix implicitly. Essentially
 * computes matrix product with provided
 * right-hand side part *twice*.
 */
struct DenseImplicitSquareMatrixOperation
{
	DenseImplicitSquareMatrixOperation(const DenseMatrix& matrix) : _matrix(matrix)
	{
	}
	/** Computes matrix product of the matrix and provided right-hand 
	 * side matrix *twice*
	 */
	inline DenseMatrix operator()(DenseMatrix operatee)
	{
		return _matrix.selfadjointView<Eigen::Upper>()*(_matrix.selfadjointView<Eigen::Upper>()*operatee);
	}
	const DenseMatrix& _matrix;
};

#endif
