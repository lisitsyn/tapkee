/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <istream>
#include <ostream>

using namespace std;

tapkee::DenseMatrix read_data(ifstream& ifs)
{
	string str;
	vector< vector<tapkee::ScalarType> > input_data;
	while (!ifs.eof())
	{
		getline(ifs,str);
		if (str.size())
		{
			stringstream strstr(str);
			istream_iterator<tapkee::ScalarType> it(strstr);
			istream_iterator<tapkee::ScalarType> end;
			vector<tapkee::ScalarType> row(it, end);
			input_data.push_back(row);
		}
	}
	tapkee::DenseMatrix fm(input_data[0].size(),input_data.size());
	for (int i=0; i<fm.rows(); i++)
	{
		for (int j=0; j<fm.cols(); j++)
			fm(i,j) = input_data[j][i];
	}
	return fm;
}

bool method_needs_kernel(tapkee::TAPKEE_METHOD method) 
{
	switch (method)
	{
#define IF_NEEDS_KERNEL(X) case X : return tapkee::MethodTraits<X>::needs_kernel(); break;
		IF_NEEDS_KERNEL(tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT);
		IF_NEEDS_KERNEL(tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING);
		IF_NEEDS_KERNEL(tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING);
		IF_NEEDS_KERNEL(tapkee::MULTIDIMENSIONAL_SCALING);
		IF_NEEDS_KERNEL(tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING);
		IF_NEEDS_KERNEL(tapkee::ISOMAP);
		IF_NEEDS_KERNEL(tapkee::LANDMARK_ISOMAP);
		IF_NEEDS_KERNEL(tapkee::DIFFUSION_MAP);
		IF_NEEDS_KERNEL(tapkee::KERNEL_PCA);
		IF_NEEDS_KERNEL(tapkee::PCA);
		IF_NEEDS_KERNEL(tapkee::RANDOM_PROJECTION);
		IF_NEEDS_KERNEL(tapkee::LAPLACIAN_EIGENMAPS);
		IF_NEEDS_KERNEL(tapkee::LOCALITY_PRESERVING_PROJECTIONS);
		IF_NEEDS_KERNEL(tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING);
		IF_NEEDS_KERNEL(tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
		IF_NEEDS_KERNEL(tapkee::STOCHASTIC_PROXIMITY_EMBEDDING);
		IF_NEEDS_KERNEL(tapkee::PASS_THRU);
		IF_NEEDS_KERNEL(tapkee::FACTOR_ANALYSIS);
		IF_NEEDS_KERNEL(tapkee::UNKNOWN_METHOD);
#undef IF_NEEDS_KERNEL
	}
	return false;
}

bool method_needs_distance(tapkee::TAPKEE_METHOD method)
{
	switch (method)
	{
#define IF_NEEDS_DISTANCE(X) case X : return tapkee::MethodTraits<X>::needs_distance(); break;
		IF_NEEDS_DISTANCE(tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT);
		IF_NEEDS_DISTANCE(tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::MULTIDIMENSIONAL_SCALING);
		IF_NEEDS_DISTANCE(tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING);
		IF_NEEDS_DISTANCE(tapkee::ISOMAP);
		IF_NEEDS_DISTANCE(tapkee::LANDMARK_ISOMAP);
		IF_NEEDS_DISTANCE(tapkee::DIFFUSION_MAP);
		IF_NEEDS_DISTANCE(tapkee::KERNEL_PCA);
		IF_NEEDS_DISTANCE(tapkee::PCA);
		IF_NEEDS_DISTANCE(tapkee::RANDOM_PROJECTION);
		IF_NEEDS_DISTANCE(tapkee::LAPLACIAN_EIGENMAPS);
		IF_NEEDS_DISTANCE(tapkee::LOCALITY_PRESERVING_PROJECTIONS);
		IF_NEEDS_DISTANCE(tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
		IF_NEEDS_DISTANCE(tapkee::STOCHASTIC_PROXIMITY_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::PASS_THRU);
		IF_NEEDS_DISTANCE(tapkee::FACTOR_ANALYSIS);
		IF_NEEDS_DISTANCE(tapkee::UNKNOWN_METHOD);
#undef IF_NEEDS_DISTANCE
	}
	return false;
}

tapkee::TAPKEE_METHOD parse_reduction_method(const char* str)
{
	if (!strcmp(str,"local_tangent_space_alignment"))
		return tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"locally_linear_embedding"))
		return tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"hessian_locally_linear_embedding"))
		return tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"multidimensional_scaling"))
		return tapkee::MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"landmark_multidimensional_scaling"))
		return tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"isomap"))
		return tapkee::ISOMAP;
	if (!strcmp(str,"landmark_isomap"))
		return tapkee::LANDMARK_ISOMAP;
	if (!strcmp(str,"diffusion_map"))
		return tapkee::DIFFUSION_MAP;
	if (!strcmp(str,"kernel_pca"))
		return tapkee::KERNEL_PCA;
	if (!strcmp(str,"pca"))
		return tapkee::PCA;
	if (!strcmp(str,"random_projection"))
		return tapkee::RANDOM_PROJECTION;
	if (!strcmp(str,"laplacian_eigenmaps"))
		return tapkee::LAPLACIAN_EIGENMAPS;
	if (!strcmp(str,"locality_preserving_projections"))
		return tapkee::LOCALITY_PRESERVING_PROJECTIONS;
	if (!strcmp(str,"neighborhood_preserving_embedding"))
		return tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING;
	if (!strcmp(str,"linear_local_tangent_space_alignment"))
		return tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"stochastic_proximity_embedding"))
		return tapkee::STOCHASTIC_PROXIMITY_EMBEDDING;
	if (!strcmp(str,"passthru"))
		return tapkee::PASS_THRU;
	if (!strcmp(str,"factor_analysis"))
		return tapkee::FACTOR_ANALYSIS;

	return tapkee::UNKNOWN_METHOD;
}

tapkee::TAPKEE_NEIGHBORS_METHOD parse_neighbors_method(const char* str)
{
	if (!strcmp(str,"brute"))
		return tapkee::BRUTE_FORCE;
	if (!strcmp(str,"covertree"))
		return tapkee::COVER_TREE;

	return tapkee::UNKNOWN_NEIGHBORS_METHOD;
}

tapkee::TAPKEE_EIGEN_EMBEDDING_METHOD parse_eigen_method(const char* str)
{
	if (!strcmp(str,"arpack"))
		return tapkee::ARPACK;
	if (!strcmp(str,"randomized"))
		return tapkee::RANDOMIZED;
	if (!strcmp(str,"dense"))
		return tapkee::EIGEN_DENSE_SELFADJOINT_SOLVER;

	return tapkee::UNKNOWN_EIGEN_METHOD;
}

template <class RandomAccessIterator, class PairwiseCallback>
tapkee::DenseMatrix matrix_from_callback(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback callback)
{
	tapkee::DenseMatrix result((end-begin),(end-begin));
	const int N = end-begin;
	unsigned int i,j;
	for (i=0; i<N; ++i)
	{
		for (j=i; j<N; j++)
		{
			tapkee::ScalarType res = callback(begin[i],begin[j]);
			result(i,j) = res;
			result(j,i) = res;
		}
	}
	return result;
}


