#include "edrt/libedrt.hpp"

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/LinearKernel.h>

#include <fstream>
#include <vector>

using namespace Eigen;
using namespace shogun;
using namespace std;

struct kernel_callback
{
	kernel_callback(CKernel* kernel) : _kernel(kernel) {};
	double operator()(int a, int b)
	{
		return _kernel->kernel(a, b);
	}
	CKernel* _kernel;
};


int main()
{
	ifstream ifs("input.dat");

	const int count = 500;
	init_shogun_with_defaults();
	SGMatrix<double> fm(25,count);
	vector<int> data_indices;
	for (int i=0; i<count; i++)
	{
		data_indices.push_back(i);
		for (int j=0; j<25; j++)
			fm(j,i) = CMath::normal_random(0.0,1.0);
	}
	CDenseFeatures<double>* features = new CDenseFeatures<double>(fm);

	CKernel* kernel = new CLinearKernel(features,features);
	kernel_callback cb(kernel);

	edrt_options_t options;
	MatrixXd embedding = embed(data_indices.begin(),data_indices.end(),cb,options,2,0,200);
	ofstream ofs("output.dat");
	ofs << embedding;
	ofs.close();
	return 0;
}
