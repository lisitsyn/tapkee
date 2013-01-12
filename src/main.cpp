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
	if (!strcmp(str,"lisomap"))
		return tapkee::LANDMARK_ISOMAP;
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
			result((i_iter-begin),(j_iter-begin)) = callback(*i_iter,*j_iter);
		}
	}
	return result;
}

int main(int argc, const char** argv)
{
	ezOptionParser opt;
	opt.footer = "Copyright (C) 2012 Sergey Lisitsyn, Fernando Iglesias\n";
	opt.overview = "Tapkee library sample application for dense matrix embedding.";
	opt.example = "Run kernel locally linear embedding with k=10 with arpack "
                  "eigensolver on data from input.dat saving embedding to output.dat \n\n"
	              "tapkee -i input.dat -o output.dat --method klle --eigen_method arpack -k 10\n";
	opt.syntax = "tapkee_app [options]\n";

	opt.add("",0,1,0,"Input file","-i","--input-file");
	opt.add("",0,1,0,"Output file","-o","--output-file");
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
	opt.add("0",0,0,0,"Check if neighborhood graph is connected (detaulf do not check)", "--check_connectivity");
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
		tapkee::LoggingSingleton::instance().enable_info();
	}

	if (opt.isSet("--benchmark"))
	{
		tapkee::LoggingSingleton::instance().enable_benchmark();
		tapkee::LoggingSingleton::instance().message_info("Benchmarking enabled");
	}
	
	tapkee::ParametersMap parameters;

	{
		string method;
		opt.get("--method")->getString(method);
		tapkee::TAPKEE_METHOD tapkee_method = parse_reduction_method(method.c_str());
		if (tapkee_method==tapkee::UNKNOWN_METHOD)
		{
			tapkee::LoggingSingleton::instance().message_error(string("Unknown method ") + method);
			return 0;
		}
		else
			parameters[tapkee::REDUCTION_METHOD] = tapkee_method;
	}
	{
		string method;
		opt.get("--neighbors_method")->getString(method);
		tapkee::TAPKEE_NEIGHBORS_METHOD tapkee_neighbors_method = parse_neighbors_method(method.c_str());
		if (tapkee_neighbors_method==tapkee::UNKNOWN_NEIGHBORS_METHOD)
		{
			tapkee::LoggingSingleton::instance().message_error(string("Unknown neighbors method ") + method);
			return 0;
		}
		else
			parameters[tapkee::NEIGHBORS_METHOD] = tapkee_neighbors_method;
	}
	{
		string method;
		opt.get("--eigen_method")->getString(method);
		tapkee::TAPKEE_EIGEN_EMBEDDING_METHOD tapkee_eigen_method = parse_eigen_method(method.c_str());
		if (tapkee_eigen_method==tapkee::UNKNOWN_EIGEN_METHOD)
		{
			tapkee::LoggingSingleton::instance().message_error(string("Unknown eigendecomposition method ") + method);
			return 0;
		}
		else
			parameters[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee_eigen_method;
	}
	{
		int target_dimension = 1;
		opt.get("--target_dimension")->getInt(target_dimension);
		if (target_dimension < 0)
		{
			tapkee::LoggingSingleton::instance().message_error("Negative target dimensionality is not possible in current circumstances. "
			                                                   "Please visit other universe");
			return 0;
		}
		else
			parameters[tapkee::TARGET_DIMENSION] = static_cast<unsigned int>(target_dimension);
	}
	{
		int k = 1;
		opt.get("--n_neighbors")->getInt(k);
		if (k < 3)
			tapkee::LoggingSingleton::instance().message_error("The provided number of neighbors is too small, consider at least 10.");
		else
			parameters[tapkee::NUMBER_OF_NEIGHBORS] = static_cast<unsigned int>(k);
	}
	{
		double width = 1.0;
		opt.get("--width")->getDouble(width);
		if (width < 0.0)
			tapkee::LoggingSingleton::instance().message_error("Width of the gaussian kernel is negative.");
		else
			parameters[tapkee::GAUSSIAN_KERNEL_WIDTH] = static_cast<tapkee::DefaultScalarType>(width);
		int timesteps = 1;
		opt.get("--timesteps")->getInt(timesteps);
		if (timesteps < 0)
			tapkee::LoggingSingleton::instance().message_error("Number of timesteps is negative.");
		else
			parameters[tapkee::DIFFUSION_MAP_TIMESTEPS] = static_cast<unsigned int>(3);
	}

	if (opt.isSet("--spe_local"))
		parameters[tapkee::SPE_GLOBAL_STRATEGY] = static_cast<bool>(false);
	else
		parameters[tapkee::SPE_GLOBAL_STRATEGY] = static_cast<bool>(true);

	if (opt.isSet("--check_connectivity"))
		parameters[tapkee::CHECK_CONNECTIVITY] = static_cast<bool>(true);
	else
		parameters[tapkee::CHECK_CONNECTIVITY] = static_cast<bool>(false);

	{
		// keep it static yet
		parameters[tapkee::SPE_TOLERANCE] = static_cast<tapkee::DefaultScalarType>(1e-5);
		parameters[tapkee::SPE_NUM_UPDATES] = static_cast<unsigned int>(100);
		parameters[tapkee::LANDMARK_RATIO] = static_cast<tapkee::DefaultScalarType>(0.2);
		parameters[tapkee::EIGENSHIFT] = static_cast<tapkee::DefaultScalarType>(1e-9);
	}

	// Load data
	string input_filename;
	string output_filename;
	if (!opt.isSet("--input-file"))
	{
		tapkee::LoggingSingleton::instance().message_error("No input file specified");
		return 0;
	}
	else
		opt.get("--input-file")->getString(input_filename);

	if (!opt.isSet("--output-file"))
	{
		tapkee::LoggingSingleton::instance().message_error("No output file specified");
		return 0;
	}
	else
		opt.get("--output-file")->getString(output_filename);

	ifstream ifs(input_filename.c_str());
	ofstream ofs(output_filename.c_str());

	tapkee::DenseMatrix input_data = read_data(ifs);
	parameters[tapkee::CURRENT_DIMENSION] = static_cast<unsigned int>(input_data.rows());
	
	std::stringstream ss;
	ss << "Data contains " << input_data.cols() << " feature vectors with dimension of " << input_data.rows();
	tapkee::LoggingSingleton::instance().message_info(ss.str());
	
	vector<int> data_indices;
	for (int i=0; i<input_data.cols(); i++)
		data_indices.push_back(i);
	// Embed
	tapkee::DenseMatrix embedding;
	
#ifdef USE_PRECOMPUTED
	tapkee::DenseMatrix distance_matrix = 
		matrix_from_callback(data_indices.begin(),data_indices.end(),distance_callback(input_data));
	precomputed_distance_callback dcb(distance_matrix);
	tapkee::DenseMatrix kernel_matrix = 
		matrix_from_callback(data_indices.begin(),data_indices.end(),kernel_callback(input_data));
	precomputed_kernel_callback kcb(kernel_matrix);
	feature_vector_callback fvcb(input_data);

	embedding = tapkee::embed(data_indices.begin(),data_indices.end(),kcb,dcb,fvcb,parameters);
#else
	distance_callback dcb(input_data);
	kernel_callback kcb(input_data);
	feature_vector_callback fvcb(input_data);

	embedding = tapkee::embed(data_indices.begin(),data_indices.end(),kcb,dcb,fvcb,parameters);
#endif
	// Save obtained data
	ofs << embedding;
	ofs.close();
	return 0;
}
