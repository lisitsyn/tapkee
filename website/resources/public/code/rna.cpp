#include <tapkee.hpp>
#include <callbacks/dummy_callbacks.hpp>

#include <numeric>
#include <functional>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace tapkee;
using namespace tapkee::keywords;

struct MatchKernelCallback
{
	ScalarType kernel(const string& l, const string& r)
	{
		return inner_product(l.begin(), l.end(),
		  r.begin(), 0, plus<int>(),
		  equal_to<string::value_type>());
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
	
	MatchKernelCallback kernel;

	TapkeeOutput result = initialize()
	  .withParameters((method=KernelLocallyLinearEmbedding,
	                   num_neighbors=30))
	  .withKernel(kernel)
	  .embedUsing(rnas);

	cout << result.embedding.transpose() << endl;

	return 0;
}
