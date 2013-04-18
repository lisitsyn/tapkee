#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>

#include "callbacks.hpp"

using namespace tapkee;
using namespace tapkee::keywords;
using std::vector;

TEST(Interface,ChainInterfaceOrder)
{
	const int N = 20;
	vector<float> indices(N);
	for (int i=0; i<N; i++) 
		indices[i] = i;

	float_kernel_callback kcb;
	float_distance_callback dcb;
	float_features_callback fcb;

	TapkeeOutput output;

	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withKernel(kcb).withFeatures(fcb).withDistance(dcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withKernel(kcb).withDistance(dcb).withFeatures(fcb)
			.embedRange(indices.begin(),indices.end()));

	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=KernelPCA))
			.withDistance(dcb).withKernel(kcb).withFeatures(fcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=PassThru))
			.withDistance(dcb).withFeatures(fcb).withKernel(kcb)
			.embedRange(indices.begin(),indices.end()));

	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withFeatures(fcb).withDistance(dcb).withKernel(kcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withFeatures(fcb).withKernel(kcb).withDistance(dcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=PassThru))
			.withFeatures(fcb).withKernel(kcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=PassThru))
			.withFeatures(fcb).withDistance(dcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=KernelPCA))
			.withKernel(kcb).withDistance(dcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=KernelPCA))
			.withKernel(kcb).withFeatures(fcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withDistance(dcb).withFeatures(fcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withDistance(dcb).withKernel(kcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=KernelPCA))
			.withKernel(kcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=MultidimensionalScaling))
			.withDistance(dcb)
			.embedRange(indices.begin(),indices.end()));
	
	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=PassThru))
			.withFeatures(fcb)
			.embedRange(indices.begin(),indices.end()));
}

TEST(Interface, NoDataPassThru) 
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	tapkee::dummy_kernel_callback<int> kcb;
	tapkee::dummy_distance_callback<int> dcb;
	tapkee::dummy_features_callback<int> fcb;

	TapkeeOutput output;
	// should produce no error
	ASSERT_NO_THROW(output = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,(method = PassThru)));
	// that's normal
	ASSERT_EQ(0,output.embedding.cols());
	// that's normal
	ASSERT_EQ(0,output.embedding.rows());
	// that's normal
	ASSERT_EQ(NULL,output.projection.implementation);
}

TEST(Interface, ParameterTargetDimension)
{
	int td = 3;
	Parameter target_dimension = Parameter::create("td", td);
	ASSERT_EQ(td,static_cast<int>(target_dimension));
}

TEST(Interface, ParametersSet)
{
	int td = 3;
	int k = 5;
	ParametersSet pg = (target_dimension=td, num_neighbors=k);
	ASSERT_EQ(static_cast<int>(pg(target_dimension)),td);
	ASSERT_EQ(static_cast<int>(pg(num_neighbors)),k);
}

TEST(Interface, OneParameterParametersSet)
{
	int td = 3;
	ParametersSet pg = (target_dimension=td);
	ASSERT_EQ(static_cast<int>(pg(target_dimension)),td);
}

TEST(Interface, WrongParameterValueKernelLocallyLinearEmbedding) 
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	tapkee::dummy_kernel_callback<int> kcb;
	tapkee::dummy_distance_callback<int> dcb;
	tapkee::dummy_features_callback<int> fcb;

	TapkeeOutput output;
	// fails with wrong parameter type as '-1' is not a valid value.
	ASSERT_THROW(output = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=KernelLocallyLinearEmbedding,num_neighbors=-3)), 
	             wrong_parameter_error);
}

TEST(Interface, MultipleParameterKernelLocallyLinearEmbedding) 
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	tapkee::dummy_kernel_callback<int> kcb;
	tapkee::dummy_distance_callback<int> dcb;
	tapkee::dummy_features_callback<int> fcb;

	TapkeeOutput output;
	// fails with wrong parameter type as '-1' is not a valid value.
	ASSERT_THROW(output = embed(data.begin(),data.end(),kcb,dcb,fcb,
	                            (method=KernelLocallyLinearEmbedding,num_neighbors=6,num_neighbors=5)), 
	             multiple_parameter_error);
}


bool always_cancel()
{
	return true;
}

TEST(Interface, CancellationPassThru)
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	tapkee::dummy_kernel_callback<int> kcb;
	tapkee::dummy_distance_callback<int> dcb;
	tapkee::dummy_features_callback<int> fcb;

	TapkeeOutput output;
	// should cancel
	ASSERT_THROW(output = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=PassThru,cancel_function=always_cancel)),
	             cancelled_exception);
}

TEST(Interface, NoReductionMethodSetFailPassThru)
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	tapkee::dummy_kernel_callback<int> kcb;
	tapkee::dummy_distance_callback<int> dcb;
	tapkee::dummy_features_callback<int> fcb;

	TapkeeOutput output;
	// should fail with missed parameter
	ASSERT_THROW(output = embed(data.begin(),data.end(),kcb,dcb,fcb,(eigen_method=Dense)),
	             missed_parameter_error);
}

TEST(Interface, UnsupportedRandomizedForGeneralizedLE)
{
	vector<int> data;
	for (int i=0; i<20; i++) 
		data.push_back(i);
	
	tapkee::dummy_kernel_callback<int> kcb;
	float_distance_callback dcb;
	tapkee::dummy_features_callback<int> fcb;

	TapkeeOutput output;
	ASSERT_THROW(output = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=LaplacianEigenmaps,eigen_method=Randomized)),
	             unsupported_method_error);
}

TEST(Interface, EigenDecompositionFailMDS)
{
	const int N = 100;
	vector<float> data(N);
	for (int i=0; i<N; i++) 
		data[i] = 0.0;
	
	tapkee::dummy_kernel_callback<float> kcb;
	float_distance_callback dcb;
	tapkee::dummy_features_callback<float> fcb;

	TapkeeOutput output;
	ASSERT_THROW(output = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,(method=MultidimensionalScaling,eigen_method=Randomized)),
	             eigendecomposition_error);
}

TEST(Interface, NotEnoughMemoryMDS)
{
	const int N = 10000000;
	vector<float> data(N);
	for (int i=0; i<N; i++) 
		data[i] = i;
	
	tapkee::dummy_kernel_callback<float> kcb;
	float_distance_callback dcb;
	tapkee::dummy_features_callback<float> fcb;

	tapkee::TapkeeOutput output;
	// tries to form 10000000 x 10000000 matrix (won't work on any machine in 2013)
	ASSERT_THROW(output = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=MultidimensionalScaling,eigen_method=Dense)),
	             not_enough_memory_error);
}
