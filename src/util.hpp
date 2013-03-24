/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias 
 */

#ifndef TAPKEE_APP_UTIL_H_
#define TAPKEE_APP_UTIL_H_

#include <istream>
#include <fstream>
#include <ostream>
#include <iterator>

using namespace std;

inline bool is_wrong_char(char c) {
	if (!(isdigit(c) || isblank(c) || c == '.' || c == '-' || c == '+' || c == 'e'))
	{
		return true;
	}
	return false;
} 

// TODO this absolutely unexceptionally definitive should be improved later
tapkee::DenseMatrix read_data(ifstream& ifs)
{
	string str;
	vector< vector<tapkee::ScalarType> > input_data;
	while (!ifs.eof())
	{
		getline(ifs,str);

		if (find_if(str.begin(), str.end(), is_wrong_char) != str.end())
			throw std::runtime_error("Input file contains some junk, please check it");

		if (str.size())
		{
			stringstream strstr(str);
			istream_iterator<tapkee::ScalarType> it(strstr);
			istream_iterator<tapkee::ScalarType> end;
			vector<tapkee::ScalarType> row(it, end);
			input_data.push_back(row);
		}
	}

	tapkee::DenseMatrix fm(input_data.size(),input_data[0].size());
	for (int i=0; i<fm.rows(); i++)
	{
		if (static_cast<tapkee::DenseMatrix::Index>(input_data[i].size()) != fm.cols()) 
		{
			stringstream ss;
			ss << "Wrong data at line " << i;
			throw std::runtime_error(ss.str());
		}
		for (int j=0; j<fm.cols(); j++)
			fm(i,j) = input_data[i][j];
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
		IF_NEEDS_KERNEL(tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING);
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
		IF_NEEDS_DISTANCE(tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING);
		IF_NEEDS_DISTANCE(tapkee::UNKNOWN_METHOD);
#undef IF_NEEDS_DISTANCE
	}
	return false;
}

tapkee::TAPKEE_METHOD parse_reduction_method(const char* str)
{
	if (!strcmp(str,"local_tangent_space_alignment") || !strcmp(str,"ltsa"))
		return tapkee::KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"locally_linear_embedding") || !strcmp(str,"lle"))
		return tapkee::KERNEL_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"hessian_locally_linear_embedding") || !strcmp(str,"hlle"))
		return tapkee::HESSIAN_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"multidimensional_scaling") || !strcmp(str,"mds"))
		return tapkee::MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"landmark_multidimensional_scaling") || !strcmp(str,"l-mds"))
		return tapkee::LANDMARK_MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"isomap"))
		return tapkee::ISOMAP;
	if (!strcmp(str,"landmark_isomap") || !strcmp(str,"l-isomap"))
		return tapkee::LANDMARK_ISOMAP;
	if (!strcmp(str,"diffusion_map") || !strcmp(str,"dm"))
		return tapkee::DIFFUSION_MAP;
	if (!strcmp(str,"kernel_pca") || !strcmp(str,"kpca"))
		return tapkee::KERNEL_PCA;
	if (!strcmp(str,"pca"))
		return tapkee::PCA;
	if (!strcmp(str,"random_projection") || !strcmp(str,"ra"))
		return tapkee::RANDOM_PROJECTION;
	if (!strcmp(str,"laplacian_eigenmaps") || !strcmp(str,"la"))
		return tapkee::LAPLACIAN_EIGENMAPS;
	if (!strcmp(str,"locality_preserving_projections") || !strcmp(str,"lpp"))
		return tapkee::LOCALITY_PRESERVING_PROJECTIONS;
	if (!strcmp(str,"neighborhood_preserving_embedding") || !strcmp(str,"npe"))
		return tapkee::NEIGHBORHOOD_PRESERVING_EMBEDDING;
	if (!strcmp(str,"linear_local_tangent_space_alignment") || !strcmp(str,"lltsa"))
		return tapkee::LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"stochastic_proximity_embedding") || !strcmp(str,"spe"))
		return tapkee::STOCHASTIC_PROXIMITY_EMBEDDING;
	if (!strcmp(str,"passthru"))
		return tapkee::PASS_THRU;
	if (!strcmp(str,"factor_analysis") || !strcmp(str,"fa"))
		return tapkee::FACTOR_ANALYSIS;
	if (!strcmp(str,"t-stochastic_neighbor_embedding") || !strcmp(str,"t-sne"))
		return tapkee::T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING;

	return tapkee::UNKNOWN_METHOD;
}

tapkee::TAPKEE_NEIGHBORS_METHOD parse_neighbors_method(const char* str)
{
	if (!strcmp(str,"brute"))
		return tapkee::BRUTE_FORCE;
#ifdef TAPKEE_USE_LGPL_COVERTREE
	if (!strcmp(str,"covertree"))
		return tapkee::COVER_TREE;
#endif

	return tapkee::UNKNOWN_NEIGHBORS_METHOD;
}

tapkee::TAPKEE_EIGEN_EMBEDDING_METHOD parse_eigen_method(const char* str)
{
#ifdef TAPKEE_WITH_ARPACK
	if (!strcmp(str,"arpack"))
		return tapkee::ARPACK;
#endif
	if (!strcmp(str,"randomized"))
		return tapkee::RANDOMIZED;
	if (!strcmp(str,"dense"))
		return tapkee::EIGEN_DENSE_SELFADJOINT_SOLVER;

	return tapkee::UNKNOWN_EIGEN_METHOD;
}

template <class PairwiseCallback>
tapkee::DenseMatrix matrix_from_callback(const tapkee::IndexType N, PairwiseCallback callback)
{
	tapkee::DenseMatrix result(N,N);
	tapkee::IndexType i,j;
#pragma omp parallel for shared(callback,result) private(j) default(none)
	for (i=0; i<N; ++i)
	{
		for (j=i; j<N; j++)
		{
			tapkee::ScalarType res = callback(i,j);
			result(i,j) = res;
			result(j,i) = res;
		}
	}
	return result;
}

#endif
