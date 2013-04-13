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

	dummy_kernel_callback<IndexType> kcb;
	my_distance_callback dcb;
	dummy_feature_vector_callback<IndexType> fvcb;

	ReturnResult result = embed(indices.begin(),indices.end(),
	                            kcb,dcb,fvcb,(method=MultidimensionalScaling,target_dimension=1));
	cout << result.first.transpose() << endl;

	return 0;
}
