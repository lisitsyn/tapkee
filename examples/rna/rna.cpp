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
using namespace tapkee::keywords;

struct match_kernel_callback
{
	ScalarType kernel(const string& l, const string& r) const
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
	
	match_kernel_callback k;

	ReturnResult result = withParameters((method=KernelLocallyLinearEmbedding,
	                                      num_neighbors=30,
	                                      target_dimension=2))
	                     .withKernel(k)
	                     .embed(rnas.begin(),rnas.end());
	cout << result.first.transpose() << endl;

	return 0;
}
