/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn, Fernando J. Iglesias García
 *
 * This code uses Any type developed by C. Diggins under Boost license, version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 */

#ifndef TAPKEE_EIGEN_CALLBACKS_H_
#define TAPKEE_EIGEN_CALLBACKS_H_

#include "../defines.hpp"

struct feature_vector_callback
{
	feature_vector_callback(const DenseMatrix& matrix) : feature_matrix(matrix) {};
	inline void operator()(int i, DenseVector& vector) const
	{
		vector = feature_matrix.col(i);
	}
	const DenseMatrix& feature_matrix;
};

struct kernel_callback
{
	kernel_callback(const DenseMatrix& matrix) : feature_matrix(matrix) {};
	inline DefaultScalarType operator()(int a, int b) const
	{
		return feature_matrix.col(a).dot(feature_matrix.col(b));
	}
	const DenseMatrix& feature_matrix;
};
TAPKEE_CALLBACK_IS_KERNEL(kernel_callback);

struct distance_callback
{
	distance_callback(const DenseMatrix& matrix) : feature_matrix(matrix) {};
	inline DefaultScalarType operator()(int a, int b) const
	{
		return (feature_matrix.col(a)-feature_matrix.col(b)).norm();
	}
	const DenseMatrix& feature_matrix;
};
TAPKEE_CALLBACK_IS_DISTANCE(distance_callback);

#endif
