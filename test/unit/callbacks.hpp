#include <cmath>

struct float_kernel_callback
{
	tapkee::ScalarType kernel(float a, float b) const
	{
		return a*b;
	}
};

struct float_distance_callback
{
	tapkee::ScalarType distance(float a, float b) const
	{
		return std::abs(a-b);
	}
};

struct float_features_callback
{
	tapkee::IndexType dimension() const
	{
		return 1;
	}
	void vector(float a, tapkee::DenseVector& v) const
	{
		v(0) = a;
	}
};
