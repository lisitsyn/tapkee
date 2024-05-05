/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

#include <tapkee/defines/types.hpp>

#include <memory>

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
    ProjectingFunction() : implementation()
    {
    }
    ProjectingFunction(ProjectionImplementation* impl) : implementation(impl)
    {
    }
    //! Projects provided vector to new space
    //! @param vec vector to be projected
    //! @return projected vector
    inline DenseVector operator()(const DenseVector& vec)
    {
        return implementation->project(vec);
    }
    std::shared_ptr<ProjectionImplementation> implementation;
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
        return proj_mat.transpose() * (vec - mean_vec);
    }

    DenseMatrix proj_mat;
    DenseVector mean_vec;
};

static ProjectingFunction unimplementedProjectingFunction()
{
    return tapkee::ProjectingFunction();
}

} // namespace tapkee
