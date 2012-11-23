/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#include <tapkee.hpp>
#include <tapkee_defines.hpp>
#include <callbacks/eigen_callbacks.hpp>
#include <callbacks/precomputed_callbacks.hpp>
#include <utils/logging.hpp>

#include "ezoptionparser.hpp"

#include <algorithm>
#include <string>
#include <istream>
#include <fstream>
#include <vector>
#include <iterator>

using namespace ez;
using namespace Eigen;
using namespace std;

DenseMatrix read_data(const string& filename)
{
	ifstream ifs(filename.c_str());
	string str;
	vector< vector<DefaultScalarType> > input_data;
	while (!ifs.eof())
	{
		getline(ifs,str);
		if (str.size())
		{
			stringstream strstr(str);
			istream_iterator<DefaultScalarType> it(strstr);
			istream_iterator<DefaultScalarType> end;
			vector<DefaultScalarType> row(it, end);
			input_data.push_back(row);
		}
	}
	DenseMatrix fm(input_data[0].size(),input_data.size());
	for (int i=0; i<fm.rows(); i++)
	{
		for (int j=0; j<fm.cols(); j++)
			fm(i,j) = input_data[j][i];
	}
	return fm;
}

TAPKEE_METHOD parse_reduction_method(const char* str)
{
	if (!strcmp(str,"kltsa"))
		return KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"klle"))
		return KERNEL_LOCALLY_LINEAR_EMBEDDING;
	if (!strcmp(str,"mds"))
		return MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"lmds"))
		return LANDMARK_MULTIDIMENSIONAL_SCALING;
	if (!strcmp(str,"isomap"))
		return ISOMAP;
	if (!strcmp(str,"diffusion_map"))
		return DIFFUSION_MAP;
	if (!strcmp(str,"kpca"))
		return KERNEL_PCA;
	if (!strcmp(str,"pca"))
		return PCA;
	if (!strcmp(str,"laplacian_eigenmaps"))
		return LAPLACIAN_EIGENMAPS;
	if (!strcmp(str,"lpp"))
		return LOCALITY_PRESERVING_PROJECTIONS;
	if (!strcmp(str,"npe"))
		return NEIGHBORHOOD_PRESERVING_EMBEDDING;
	if (!strcmp(str,"lltsa"))
		return LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT;
	if (!strcmp(str,"spe"))
		return STOCHASTIC_PROXIMITY_EMBEDDING;
	if (!strcmp(str,"lisomap"))
		return LANDMARK_ISOMAP;

	return UNKNOWN_METHOD;
}

TAPKEE_NEIGHBORS_METHOD parse_neighbors_method(const char* str)
{
	if (!strcmp(str,"brute"))
		return BRUTE_FORCE;
	if (!strcmp(str,"covertree"))
		return COVER_TREE;

	return UNKNOWN_NEIGHBORS_METHOD;
}

TAPKEE_EIGEN_EMBEDDING_METHOD parse_eigen_method(const char* str)
{
	if (!strcmp(str,"arpack"))
		return ARPACK;
	if (!strcmp(str,"randomized"))
		return RANDOMIZED;
	if (!strcmp(str,"dense"))
		return EIGEN_DENSE_SELFADJOINT_SOLVER;

	printf("Method %s is not supported (yet?)\n",str);
	exit(EXIT_FAILURE);
	return RANDOMIZED;
}

template <class RandomAccessIterator, class PairwiseCallback>
DenseMatrix matrix_from_callback(RandomAccessIterator begin, RandomAccessIterator end, PairwiseCallback callback)
{
	DenseMatrix result((end-begin),(end-begin));
	for (RandomAccessIterator i_iter=begin; i_iter!=end; ++i_iter)
	{
		for (RandomAccessIterator j_iter=begin; j_iter!=end; ++j_iter)
		{
			result((i_iter-begin),(j_iter-begin)) = callback(*i_iter,*j_iter);
		}
	}
	return result;
}

int main(int argc, const char** argv)
{
	ezOptionParser opt;
	opt.footer = "\n";
	opt.overview = "Tapkee library sample application for dense matrix embedding.";
	opt.example = "Run kernel locally linear embedding with k=10 with arpack "
                  "eigensolver on some dataset \n\n"
	              "tapkee --method klle -em arpack -k 10";
	opt.syntax = "tapkee_app [options]";

	opt.add("",0,0,0,"Display help","-h","--help");
	opt.add("",0,0,0,"Output benchmark information","--benchmark");
	opt.add("",0,0,0,"Output more information","--verbose");
	opt.add("klle",0,1,0,
			"Dimension reduction method (default klle). One of the following: "
			"klle, npe, kltsa, lltsa, hlle, laplacian_eigenmaps, lpp, "
			"diffusion_map, isomap, lisomap, mds, lmds, spe, kpca, pca.",
			"-m","--method");
	opt.add("covertree",0,1,0,
			"Neighbors search method (default covertree). One of the following "
			"covertree, brute.",
			"-nm","--neighbors_method");
	opt.add("arpack",0,1,0,
			"Eigendecomposition method (default arpack). One of the following "
			"arpack, randomized, dense.",
			"-em","--eigen_method");
	opt.add("2",0,1,0,"Target dimension (default 2)","-td","--target_dimension");
	opt.add("10",0,1,0,"Number of neighbors (default 10)","-nn","-k","--n_neighbors");
	opt.add("1.0",0,1,0,"Width of gaussian kernel (default 1.0)","-w","--width");
	opt.add("1",0,1,0,"Number of timesteps for diffusion map (default 1)","--timesteps");
	opt.add("0",0,0,0,"Local strategy in SPE (default global)", "--spe_local");
	opt.parse(argc, argv);

	if (opt.isSet("-h"))
	{
		string usage;
		opt.getUsage(usage);
		std::cout << usage << std::endl;
		return 0;
	}

	if (opt.isSet("--verbose"))
	{
		LoggingSingleton::instance().enable_info();
	}

	if (opt.isSet("--benchmark"))
	{
		LoggingSingleton::instance().enable_benchmark();
		LoggingSingleton::instance().message_info("Benchmarking enabled");
	}
	
	ParametersMap parameters;

	{
		string method;
		opt.get("--method")->getString(method);
		TAPKEE_METHOD tapkee_method = parse_reduction_method(method.c_str());
		if (tapkee_method==UNKNOWN_METHOD)
		{
			LoggingSingleton::instance().message_error(string("Unknown method ") + method);
			return 0;
		}
		else
			parameters[REDUCTION_METHOD] = tapkee_method;
	}
	{
		string method;
		opt.get("--neighbors_method")->getString(method);
		TAPKEE_NEIGHBORS_METHOD tapkee_neighbors_method = parse_neighbors_method(method.c_str());
		if (tapkee_neighbors_method==UNKNOWN_NEIGHBORS_METHOD)
		{
			LoggingSingleton::instance().message_error(string("Unknown neighbors method ") + method);
			return 0;
		}
		else
			parameters[NEIGHBORS_METHOD] = tapkee_neighbors_method;
	}
	{
		string method;
		opt.get("--eigen_method")->getString(method);
		TAPKEE_EIGEN_EMBEDDING_METHOD tapkee_eigen_method = parse_eigen_method(method.c_str());
		if (tapkee_eigen_method==UNKNOWN_EIGEN_METHOD)
		{
			LoggingSingleton::instance().message_error(string("Unknown eigendecomposition method ") + method);
			return 0;
		}
		else
			parameters[EIGEN_EMBEDDING_METHOD] = tapkee_eigen_method;
	}
	{
		int target_dimension = 1;
		opt.get("--target_dimension")->getInt(target_dimension);
		if (target_dimension < 0)
		{
			LoggingSingleton::instance().message_error("Negative target dimensionality is not possible in current circumstances. "
			                                           "Please visit other universe");
			return 0;
		}
		else
			parameters[TARGET_DIMENSION] = static_cast<unsigned int>(target_dimension);
	}
	{
		int k = 1;
		opt.get("--n_neighbors")->getInt(k);
		if (k < 3)
			LoggingSingleton::instance().message_error("The provided number of neighbors is too small, consider at least 10.");
		else
			parameters[NUMBER_OF_NEIGHBORS] = static_cast<unsigned int>(k);
	}
	{
		double width = 1.0;
		opt.get("--width")->getDouble(width);
		if (width < 0.0)
			LoggingSingleton::instance().message_error("Width of the gaussian kernel is negative.");
		else
			parameters[GAUSSIAN_KERNEL_WIDTH] = static_cast<DefaultScalarType>(width);
		int timesteps = 1;
		opt.get("--timesteps")->getInt(timesteps);
		if (timesteps < 0)
			LoggingSingleton::instance().message_error("Number of timesteps is negative.");
		else
			parameters[DIFFUSION_MAP_TIMESTEPS] = static_cast<unsigned int>(3);
	}

	if (opt.isSet("--spe_local"))
		parameters[SPE_GLOBAL_STRATEGY] = static_cast<bool>(false);
	else
		parameters[SPE_GLOBAL_STRATEGY] = static_cast<bool>(true);

	{
		// keep it static yet
		parameters[DIFFUSION_MAP_TIMESTEPS] = static_cast<unsigned int>(3);
		parameters[GAUSSIAN_KERNEL_WIDTH] = static_cast<DefaultScalarType>(1000.0);
		parameters[SPE_TOLERANCE] = static_cast<DefaultScalarType>(1e-5);
		parameters[SPE_NUM_UPDATES] = static_cast<unsigned int>(100);
		parameters[LANDMARK_RATIO] = static_cast<DefaultScalarType>(0.2);
		parameters[EIGENSHIFT] = static_cast<DefaultScalarType>(1e-9);
	}

	// Load data
	DenseMatrix input_data = read_data("input.dat");
	parameters[CURRENT_DIMENSION] = static_cast<unsigned int>(input_data.rows());
	
	std::stringstream ss;
	ss << "Data contains " << input_data.cols() << " feature vectors with dimension of " << input_data.rows();
	LoggingSingleton::instance().message_info(ss.str());
	
	vector<int> data_indices;
	for (int i=0; i<input_data.cols(); i++)
		data_indices.push_back(i);
	// Embed
	DenseMatrix embedding;
	
#ifdef USE_PRECOMPUTED
	DenseMatrix distance_matrix = 
		matrix_from_callback(data_indices.begin(),data_indices.end(),distance_callback(input_data));
	precomputed_distance_callback dcb(distance_matrix);
	DenseMatrix kernel_matrix = 
		matrix_from_callback(data_indices.begin(),data_indices.end(),kernel_callback(input_data));
	precomputed_kernel_callback kcb(kernel_matrix);
	feature_vector_callback fvcb(input_data);

	embedding = embed(data_indices.begin(),data_indices.end(),kcb,dcb,fvcb,parameters);
#else
	distance_callback dcb(input_data);
	kernel_callback kcb(input_data);
	feature_vector_callback fvcb(input_data);

	embedding = embed(data_indices.begin(),data_indices.end(),kcb,dcb,fvcb,parameters);
#endif
	// Save obtained data
	ofstream ofs("output.dat");
	ofs << embedding;
	ofs.close();
	return 0;
}
