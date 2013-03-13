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
		return abs(a-b);
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
