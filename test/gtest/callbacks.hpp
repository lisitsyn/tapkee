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

struct float_features_callback
{
	void vector(float a, tapkee::DenseVector& v)
	{
		v(0) = a;
	}
};
