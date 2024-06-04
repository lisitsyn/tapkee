#include <tapkee/callbacks/precomputed_callbacks.hpp>
#include <tapkee/tapkee.hpp>

using namespace std;
using namespace tapkee;

int main(int argc, const char **argv)
{
    const int N = 100;
    DenseMatrix distances(N, N);
    vector<IndexType> indices(N);
    for (int i = 0; i < N; i++)
    {
        indices[i] = i;

        for (int j = 0; j < N; j++)
            distances(i, j) = abs(i - j);
    }

    precomputed_distance_callback distance(distances);

    TapkeeOutput output = with((method = MultidimensionalScaling, target_dimension = 1))
                              .withDistance(distance)
                              .embedUsing(indices);

    cout << output.embedding.transpose() << endl;

    return 0;
}
