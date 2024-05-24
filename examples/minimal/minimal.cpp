#include <tapkee/tapkee.hpp>

using namespace std;
using namespace tapkee;

struct MyDistanceCallback
{
    ScalarType distance(IndexType l, IndexType r)
    {
        return abs(static_cast<ScalarType>(l) - static_cast<ScalarType>(r));
    }
};

int main(int argc, const char **argv)
{
    const int N = 100;
    vector<IndexType> indices(N);
    for (int i = 0; i < N; i++)
        indices[i] = i;

    MyDistanceCallback distance;

    TapkeeOutput output = with((method = MultidimensionalScaling, target_dimension = 1))
                              .withDistance(distance)
                              .embedUsing(indices);

    cout << output.embedding.transpose() << endl;

    return 0;
}
