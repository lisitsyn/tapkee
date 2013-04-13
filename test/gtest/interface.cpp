#include <gtest/gtest.h>

#include <tapkee.hpp>
#include <tapkee_exceptions.hpp>

#include "callbacks.hpp"

using namespace tapkee;
using namespace tapkee::keywords;
using std::vector;

TEST(Interface, NoDataPassThru) 
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	// should produce no error
	ASSERT_NO_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,(method = PassThru)));
	// that's normal
	ASSERT_EQ(0,result.first.cols());
	// that's normal
	ASSERT_EQ(0,result.first.rows());
	// that's normal
	ASSERT_EQ(NULL,result.second.implementation);
}

TEST(Interface, ParameterTargetDimension)
{
	int td = 3;
	Parameter target_dimension = Parameter::create("td", td);
	ASSERT_EQ(td,static_cast<int>(target_dimension));
}

TEST(Interface, ParameterGroup)
{
	int td = 3;
	int k = 5;
	ParameterGroup pg = (target_dimension=td, num_neighbors=k);
	ASSERT_EQ(static_cast<int>(pg(target_dimension)),td);
	ASSERT_EQ(static_cast<int>(pg(num_neighbors)),k);
}

TEST(Interface, OneParameterParameterGroup)
{
	int td = 3;
	ParameterGroup pg = (target_dimension=td);
	ASSERT_EQ(static_cast<int>(pg(target_dimension)),td);
}

TEST(Interface, WrongParameterValueKernelLocallyLinearEmbedding) 
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	// fails with wrong parameter type as '-1' is not a valid value.
	ASSERT_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=KernelLocallyLinearEmbedding,num_neighbors=-3)), 
	             wrong_parameter_error);
}

TEST(Interface, MultipleParameterKernelLocallyLinearEmbedding) 
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	// fails with wrong parameter type as '-1' is not a valid value.
	ASSERT_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,
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
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	// should cancel
	ASSERT_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=PassThru,cancel_function=always_cancel)),
	             cancelled_exception);
}

TEST(Interface, NoReductionMethodSetFailPassThru)
{
	vector<int> data;
	ASSERT_EQ(0,data.size());
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	// should fail with missed parameter
	ASSERT_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,(eigen_method=Dense)),
	             missed_parameter_error);
}

TEST(Interface, UnsupportedRandomizedForGeneralizedLE)
{
	vector<int> data;
	for (int i=0; i<20; i++) 
		data.push_back(i);
	
	dummy_kernel_callback kcb;
	float_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	ASSERT_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=LaplacianEigenmaps,eigen_method=Randomized)),
	             unsupported_method_error);
}

TEST(Interface, EigenDecompositionFailMDS)
{
	vector<int> data;
	for (int i=0; i<20; i++) 
		data.push_back(i);
	
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	ReturnResult result;
	ASSERT_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,(method=MultidimensionalScaling,eigen_method=Randomized)),
	             eigendecomposition_error);
}

TEST(Interface, NotEnoughMemoryMDS)
{
	vector<int> data;
	for (int i=0; i<10000000; i++) 
		data.push_back(i);
	
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;

	tapkee::ReturnResult result;
	// tries to form 10000000 x 10000000 matrix (won't work on any machine in 2013)
	ASSERT_THROW(result = embed(data.begin(),data.end(),kcb,dcb,fcb,(method=MultidimensionalScaling,eigen_method=Dense)),
	             not_enough_memory_error);
}
