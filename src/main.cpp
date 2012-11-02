/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#include "edrt.hpp"
#include "utils/time.hpp"

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/LinearKernel.h>
#include <algorithm>
#include <string>
#include <istream>
#include <fstream>
#include <vector>
#include <iterator>

using namespace Eigen;
using namespace shogun;
using namespace std;

struct kernel_callback
{
	kernel_callback(CKernel* kernel) : _kernel(kernel) {};
	double operator()(int a, int b) const
	{
		return _kernel->kernel(a, b);
	}
	CKernel* _kernel;
};
vector< vector<double> > read_data(const string& filename)
{
	ifstream ifs(filename.c_str());
	string str;
	vector< vector<double> > input_data;
	while (!ifs.eof())
	{
		getline(ifs,str);
		if (str.size())
		{
			stringstream strstr(str);
			istream_iterator<double> it(strstr);
			istream_iterator<double> end;
			vector<double> row(it, end);
			input_data.push_back(row);
		}
	}
	return input_data;
}

int main(int argc, const char** argv)
{
	if (argc!=2)
	{
		printf("No parameters specified.\n");
		exit(EXIT_FAILURE);
	}

	ParametersMap parameters;
	parameters[REDUCTION_METHOD] = KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT;
	parameters[NEIGHBORS_METHOD] = COVER_TREE;
	parameters[EIGEN_EMBEDDING_METHOD] = RANDOMIZED_INVERSE;
	parameters[NUMBER_OF_NEIGHBORS] = static_cast<unsigned int>(atoi(argv[1]));
	parameters[TARGET_DIMENSIONALITY] = static_cast<unsigned int>(2);

	// Load data
	vector< vector<double> > input_data = read_data("input.dat");

	// Shogun part
	init_shogun_with_defaults();
	SGMatrix<double> fm(input_data[0].size(),input_data.size());
	vector<int> data_indices;
	for (int i=0; i<fm.num_rows; i++)
	{
		for (int j=0; j<fm.num_cols; j++)
			fm(i,j) = input_data[j][i];
	}
	for (int i=0; i<fm.num_cols; i++)
		data_indices.push_back(i);
	CDenseFeatures<double>* features = new CDenseFeatures<double>(fm);
	features = new CDenseFeatures<double>(fm);
	CKernel* kernel = new CLinearKernel(features,features);
	kernel_callback cb(kernel);

	// Embed
	DenseMatrix embedding = embed(data_indices.begin(),data_indices.end(),cb,parameters);

	// Save obtained data
	ofstream ofs("output.dat");
	ofs << embedding;
	ofs.close();
	return 0;
}
