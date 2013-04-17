#include <tapkee.hpp>
#include <callback/dummy_callbacks.hpp>

using namespace std;
using namespace tapkee;
using namespace tapkee::keywords;

struct my_distance_callback
{
	ScalarType distance(IndexType l, IndexType r) { return abs(l-r); } 
}; 

int main(int argc, const char** argv)
{
	const int N = 100;
	vector<IndexType> indices(N);
	for (int i=0; i<N; i++) indices[i] = i;

	my_distance_callback d;

	ReturnResult result = withParameters((method=MultidimensionalScaling,target_dimension=1))
	                     .withDistance(d)
	                     .embed(indices.begin(),indices.end());
	
	cout << result.first.transpose() << endl;

	return 0;
}
