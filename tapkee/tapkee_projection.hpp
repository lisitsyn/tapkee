/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando J. Iglesias García
 *
 */

namespace tapkee
{

//! A base class for implementation of projecting 
struct ProjectionImplementation
{
	virtual ~ProjectionImplementation()
	{
	}
	//! Projects provided vector to new space
	//! @param vec vector to be projected
	//! @return projected vector
	virtual DenseVector project(const DenseVector& vec) = 0;
};

//! A pimpl wrapper for projecting function
struct ProjectingFunction
{
	ProjectingFunction() : implementation(NULL) {};
	ProjectingFunction(ProjectionImplementation* impl) : implementation(impl) {};
	//! Destroys current implementation
	void clear() 
	{ 
		delete implementation;
	}
	//! Projects provided vector to new space
	//! @param vec vector to be projected
	//! @return projected vector
	inline DenseVector operator()(const DenseVector& vec)
	{
		return implementation->project(vec);
	}
	ProjectionImplementation* implementation;
};

//! Basic @ref ProjectionImplementation that subtracts mean from the vector
//! and multiplies projecting matrix with it.
struct MatrixProjectionImplementation : public ProjectionImplementation
{
	MatrixProjectionImplementation(DenseMatrix matrix, DenseVector mean) : proj_mat(matrix), mean_vec(mean)
	{
	}

	virtual ~MatrixProjectionImplementation()
	{
	}

	virtual DenseVector project(const DenseVector& vec) 
	{
		return proj_mat*(vec-mean_vec);
	}

	DenseMatrix proj_mat;
	DenseVector mean_vec;
};


}

