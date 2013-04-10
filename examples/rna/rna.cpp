#define TAPKEE_WITH_ARPACK
#include <tapkee.hpp>
#include <callback/dummy_callbacks.hpp>

#include <numeric>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace tapkee;

struct hamming_distance_callback
{
	ScalarType distance(const string& l, const string& r) 
	{
		return inner_product(l.begin(),l.end(),r.begin(),
		                     0,plus<int>(),not2(equal_to<string::value_type>()));
	} 
}; 

int main(int argc, const char** argv)
{
	vector<string> rnas;
	ifstream input_stream;
	input_stream.open(argv[1]);

	string line;
	while (!input_stream.eof()) {
		input_stream >> line;
		rnas.push_back(line);
	}
	
	dummy_kernel_callback<string> kcb;
	hamming_distance_callback dcb;
	dummy_feature_vector_callback<string> fvcb;

	ParametersMap parameters;
	parameters[ReductionMethod] = MultidimensionalScaling;
	parameters[TargetDimension] = static_cast<IndexType>(2);
	parameters[DiffusionMapTimesteps] = static_cast<IndexType>(1);
	parameters[GaussianKernelWidth] = static_cast<ScalarType>(10.0);

	ReturnResult result = embed(rnas.begin(),rnas.end(),
	                            kcb,dcb,fvcb,parameters);
	cout << result.first.transpose() << endl;

	return 0;
}
