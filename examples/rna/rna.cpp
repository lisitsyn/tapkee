#include <tapkee/callbacks/dummy_callbacks.hpp>
#include <tapkee/tapkee.hpp>

#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>

using namespace std;
using namespace tapkee;

struct MatchKernelCallback
{
    ScalarType kernel(const string &l, const string &r)
    {
        return inner_product(l.begin(), l.end(), r.begin(), 0, plus<int>(), equal_to<string::value_type>());
    }
};

int main(int argc, const char **argv)
{
    vector<string> rnas;
    ifstream input_stream;
    input_stream.open(argv[1]);

    string line;
    while (!input_stream.eof())
    {
        input_stream >> line;
        rnas.push_back(line);
    }

    MatchKernelCallback kernel;

    TapkeeOutput result = with((method = KernelLocallyLinearEmbedding, num_neighbors = 30))
                         .with(kernel)
                         .embedUsing(rnas);

    cout << result.embedding.transpose() << endl;

    return 0;
}
