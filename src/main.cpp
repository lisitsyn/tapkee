/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <tapkee.hpp>
#include <tapkee_defines.hpp>
#include <callbacks/eigen_callbacks.hpp>
#include <callbacks/precomputed_callbacks.hpp>
#include <utils/logging.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include <iterator>

#include "ezoptionparser.hpp"
#include "util.hpp"

#ifdef GIT_INFO
	#define TAPKEE_CURRENT_GIT_INFO GIT_INFO
#else
	#define TAPKEE_CURRENT_GIT_INFO "unknown"
#endif

using namespace ez;
using namespace Eigen;
using namespace std;

int main(int argc, const char** argv)
{
	ezOptionParser opt;
	opt.footer = "Copyright (C) 2012-2013 Sergey Lisitsyn, Fernando Iglesias\n";
	opt.overview = "Tapkee library application for reduction dimensions of dense matrices.\n"
	               "Git " TAPKEE_CURRENT_GIT_INFO;
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
	opt.add("1e-9",0,1,0,"Regularization diagonal shift for weight matrix","--eigenshift");
	opt.add("0.2",0,1,0,"Ratio of landmarks. Should be in (0,1) range","--landmark_ratio");
	opt.add("1e-5",0,1,0,"Tolerance for SPE","--spe_tolerance");
	opt.add("100",0,1,0,"Number of SPE updates","--spe_num_updates");
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
			parameters[tapkee::DIFFUSION_MAP_TIMESTEPS] = static_cast<unsigned int>(timesteps);
	}
	{
		double eigenshift = 1e-9;
		opt.get("--eigenshift")->getDouble(eigenshift);
		parameters[tapkee::EIGENSHIFT] = static_cast<tapkee::DefaultScalarType>(eigenshift);
	}
	{
		double landmark_ratio = 0.0;
		opt.get("--landmark_ratio")->getDouble(landmark_ratio);
		parameters[tapkee::LANDMARK_RATIO] = static_cast<tapkee::DefaultScalarType>(landmark_ratio);
	}
	{
		if (opt.isSet("--spe_local"))
			parameters[tapkee::SPE_GLOBAL_STRATEGY] = static_cast<bool>(false);
		else
			parameters[tapkee::SPE_GLOBAL_STRATEGY] = static_cast<bool>(true);
	}
	{
		if (opt.isSet("--check_connectivity"))
			parameters[tapkee::CHECK_CONNECTIVITY] = static_cast<bool>(true);
		else
			parameters[tapkee::CHECK_CONNECTIVITY] = static_cast<bool>(false);
	}
	{
		double spe_tolerance = 1e-5;
		opt.get("--spe_tolerance")->getDouble(spe_tolerance);
		parameters[tapkee::SPE_TOLERANCE] = static_cast<tapkee::DefaultScalarType>(spe_tolerance);
	}
	{
		int spe_num_updates = 100;
		opt.get("--spe_num_updates")->getInt(spe_num_updates);
		parameters[tapkee::SPE_NUM_UPDATES] = static_cast<unsigned int>(spe_num_updates);
	}

	// Load data
	string input_filename;
	string output_filename;
	if (!opt.isSet("--input-file"))
	{
		tapkee::LoggingSingleton::instance().message_error("No input file specified. Please use -h flag if stucked");
		return 0;
	}
	else
		opt.get("--input-file")->getString(input_filename);

	if (!opt.isSet("--output-file"))
	{
		tapkee::LoggingSingleton::instance().message_warning("No output file specified, using /dev/tty");
		output_filename = "/dev/tty";
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
	
	tapkee::DenseMatrix embedding;
	
#ifdef USE_PRECOMPUTED
	tapkee::DenseMatrix distance_matrix;
	tapkee::DenseMatrix kernel_matrix;
	{
		tapkee::timed_context context("Distance matrix computation");
		tapkee::TAPKEE_METHOD method = parameters[tapkee::REDUCTION_METHOD].cast<tapkee::TAPKEE_METHOD>();
		if (method_needs_kernel(method))
		{
			distance_matrix = 
				matrix_from_callback(data_indices.begin(),data_indices.end(),distance_callback(input_data));
		} 
		if (method_needs_distance(method))
		{
			kernel_matrix = 
				matrix_from_callback(data_indices.begin(),data_indices.end(),kernel_callback(input_data));
		}
	}
	precomputed_distance_callback dcb(distance_matrix);
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
