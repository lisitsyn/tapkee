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
	vector< vector<tapkee::DefaultScalarType> > input_data;
	while (!ifs.eof())
	{
		getline(ifs,str);
		if (str.size())
		{
			stringstream strstr(str);
			istream_iterator<tapkee::DefaultScalarType> it(strstr);
			istream_iterator<tapkee::DefaultScalarType> end;
			vector<tapkee::DefaultScalarType> row(it, end);
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
		IF_NEEDS_KERNEL(tapkee::LAPLACIAN_EIGENMAPS);
		IF_NEEDS_KERNEL(tapkee::LOCALITY_PRESERVING_PROJECTIONS);
		IF_NEEDS_KERNEL(tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING);
		IF_NEEDS_KERNEL(tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
		IF_NEEDS_KERNEL(tapkee::STOCHASTIC_PROXIMITY_EMBEDDING);
		IF_NEEDS_KERNEL(tapkee::PASS_THRU);
		IF_NEEDS_KERNEL(tapkee::UNKNOWN_METHOD);
#undef IF_NEEDS_KERNEL
	}
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
		IF_NEEDS_DISTANCE(tapkee::LAPLACIAN_EIGENMAPS);
		IF_NEEDS_DISTANCE(tapkee::LOCALITY_PRESERVING_PROJECTIONS);
		IF_NEEDS_DISTANCE(tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT);
		IF_NEEDS_DISTANCE(tapkee::STOCHASTIC_PROXIMITY_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::PASS_THRU);
		IF_NEEDS_DISTANCE(tapkee::UNKNOWN_METHOD);
#undef IF_NEEDS_DISTANCE
	}
}

tapkee::TAPKEE_METHOD parse_reduction_method(const char* str)
{
	if (!strcmp(str,"kltsa"))
		return tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"klle"))
		return tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"hlle"))
		return tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"mds"))
		return tapkee::MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"lmds"))
		return tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"isomap"))
		return tapkee::ISOMAP;
	if (!strcmp(str,"lisomap"))
		return tapkee::LANDMARK_ISOMAP;
	if (!strcmp(str,"diffusion_map"))
		return tapkee::DIFFUSION_MAP;
	if (!strcmp(str,"kpca"))
		return tapkee::KERNEL_PCA;
	if (!strcmp(str,"pca"))
		return tapkee::PCA;
	if (!strcmp(str,"laplacian_eigenmaps"))
		return tapkee::LAPLACIAN_EIGENMAPS;
	if (!strcmp(str,"lpp"))
		return tapkee::LOCALITY_PRESERVING_PROJECTIONS;
	if (!strcmp(str,"npe"))
		return tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING;
	if (!strcmp(str,"lltsa"))
		return tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"spe"))
		return tapkee::STOCHASTIC_PROXIMITY_EMBEDDING;
	if (!strcmp(str,"passthru"))
		return tapkee::PASS_THRU;

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
	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=begin; j_iter!=end; ++j_iter)
		{
			tapkee::DefaultScalarType res = callback(*i_iter,*j_iter);
			result((i_iter-begin),(j_iter-begin)) = res;
			result((j_iter-begin),(i_iter-begin)) = res;
		}
	}
	return result;
}


