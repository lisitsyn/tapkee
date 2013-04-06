struct dummy_kernel_callback 
{
	tapkee::ScalarType kernel(int, int) 
	{
		return 0.0;
	}
};
struct dummy_distance_callback 
{
	tapkee::ScalarType distance(int, int) 
	{
		return 0.0;
	}
};
struct dummy_feature_callback
{
	void vector(int, tapkee::DenseVector&)
	{
	}
};

struct float_kernel_callback
{
	tapkee::ScalarType kernel(float a, float b)
	{
		return a*b;
	}
};

struct float_distance_callback
{
	tapkee::ScalarType distance(float a, float b)
	{
		return abs(a-b);
	}
};

struct float_feature_callback
{
	void vector(float a, tapkee::DenseVector& v)
	{
		v(0) = a;
	}
};
