#include <tapkee.hpp>
#include <callback/dummy_callbacks.hpp>

using namespace std;
using namespace tapkee;

struct my_distance_callback
{
	ScalarType operator()(IndexType l, IndexType r) { return abs(l-r); } 
}; 
TAPKEE_CALLBACK_IS_DISTANCE(my_distance_callback);

int main(int argc, const char** argv)
{
	const int N = 100;
	vector<tapkee::IndexType> indices(N);
	for (int i=0; i<N; i++) indices[i] = i;

	dummy_kernel_callback<IndexType> kcb;
	my_distance_callback dcb;
	dummy_feature_vector_callback<IndexType> fvcb;

	ParametersMap parameters;
	parameters[REDUCTION_METHOD] = MULTIDIMENSIONAL_SCALING;
	parameters[TARGET_DIMENSION] = static_cast<IndexType>(1);

	ReturnResult result = embed(indices.begin(),indices.end(),
	                            kcb,dcb,fvcb,parameters);
	cout << result.first.transpose() << endl;

	return 0;
}
