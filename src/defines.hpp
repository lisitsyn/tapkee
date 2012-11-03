/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 *
 * This code uses Any type developed by C. Diggins under Boost license, version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 */

#ifndef EDRT_DEFINES_H_
#define EDRT_DEFINES_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include "utils/any.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

enum EDRT_NEIGHBORS_METHOD
{
	BRUTE_FORCE,
	COVER_TREE
};

enum EDRT_EIGEN_EMBEDDING_METHOD
{
	ARPACK_XSXUPD,
	LAPACK_XSYEVR,
	RANDOMIZED_INVERSE
};

enum EDRT_METHOD
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

enum EDRT_PARAMETERS
{
	REDUCTION_METHOD,
	NUMBER_OF_NEIGHBORS,
	TARGET_DIMENSIONALITY,
	CURRENT_DIMENSIONALITY,
	EIGEN_EMBEDDING_METHOD,
	NEIGHBORS_METHOD
};

typedef Eigen::Triplet<double> SparseTriplet;
typedef std::vector<SparseTriplet> SparseTriplets;
typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::VectorXd DenseVector;
typedef std::vector<int> LocalNeighbors;
typedef std::vector<LocalNeighbors> Neighbors;
typedef Eigen::SparseMatrix<double> SparseWeightMatrix;
typedef std::pair<Eigen::MatrixXd,Eigen::VectorXd> EmbeddingResult;
typedef std::map<EDRT_PARAMETERS, any> ParametersMap;

#endif
