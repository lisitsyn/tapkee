#include <gtest/gtest.h>

#include <tapkee.hpp>

struct dummy_kernel_callback 
{
	tapkee::ScalarType operator()(int, int) 
	{
		return 0.0;
	}
};
struct dummy_distance_callback 
{
	tapkee::ScalarType operator()(int, int) 
	{
		return 0.0;
	}
};
struct dummy_feature_callback
{
	void operator()(int i, const tapkee::DenseVector&)
	{
	}
};

TAPKEE_CALLBACK_IS_DISTANCE(dummy_distance_callback);
TAPKEE_CALLBACK_IS_KERNEL(dummy_kernel_callback);

TEST(NoDataTest, PassThru) 
{
	std::vector<int> data;
	ASSERT_EQ(0,data.size());
	dummy_kernel_callback kcb;
	dummy_distance_callback dcb;
	dummy_feature_callback fcb;
	tapkee::ParametersMap params;
	params[tapkee::REDUCTION_METHOD] = tapkee::PASS_THRU;
	params[tapkee::CURRENT_DIMENSION] = static_cast<tapkee::IndexType>(1);

	tapkee::ReturnResult result;
	ASSERT_NO_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,params));
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(0,result.first.rows());
	ASSERT_EQ(NULL,result.second.implementation);
}

struct float_kernel_callback
{
	tapkee::ScalarType operator()(float a, float b)
	{
		return a*b;
	}
};

struct float_distance_callback
{
	tapkee::ScalarType operator()(float a, float b)
	{
		return a-b;
	}
};

struct float_feature_callback
{
	void operator()(float a, tapkee::DenseVector& v)
	{
		v(0) = a;
	}
};

TAPKEE_CALLBACK_IS_DISTANCE(float_distance_callback);
TAPKEE_CALLBACK_IS_KERNEL(float_kernel_callback);

TEST(FloatCallbacks, MultidimensionalScaling)
{
	std::vector<float> data;
	data.push_back(-1.0);
	data.push_back(1.0);

	float_kernel_callback kcb;
	float_distance_callback dcb;
	float_feature_callback fcb;
	
	tapkee::ParametersMap params;
	params[tapkee::REDUCTION_METHOD] = tapkee::MULTIDIMENSIONAL_SCALING;
	params[tapkee::TARGET_DIMENSION] = static_cast<tapkee::IndexType>(1);
	params[tapkee::EIGEN_EMBEDDING_METHOD] = tapkee::EIGEN_DENSE_SELFADJOINT_SOLVER;
	params[tapkee::CURRENT_DIMENSION] = static_cast<tapkee::IndexType>(1);

	tapkee::ReturnResult result;
	ASSERT_NO_THROW(result = tapkee::embed(data.begin(),data.end(),kcb,dcb,fcb,params));
	ASSERT_EQ(1,result.first.cols());
	ASSERT_EQ(2,result.first.rows());
}
