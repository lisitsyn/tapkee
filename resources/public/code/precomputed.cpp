#include <tapkee.hpp>
#include <callbacks/precomputed_callbacks.hpp>

using namespace std;
using namespace tapkee;
using namespace tapkee::keywords;

int main(int argc, const char** argv)
{
	const int N = 100;
	tapkee::DenseMatrix distances(N,N);
	vector<IndexType> indices(N);
	for (int i=0; i<N; i++)
	{
		indices[i] = i;
		
		for (int j=0; j<N; j++)
			distances(i,j) = abs(i-j);
	}

	precomputed_distance_callback distance(distances);

	TapkeeOutput output = initialize() 
	  .withParameters((method=MultidimensionalScaling,
	                   target_dimension=1))
	  .withDistance(distance)
	  .embedUsing(indices);
	
	cout << output.embedding.transpose() << endl;

	return 0;
}

