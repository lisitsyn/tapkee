#include <gtest/gtest.h>

#include <tapkee/tapkee.hpp>

#include "data.hpp"

using namespace tapkee;

TEST(Projecting,PCA)
{
	const int N = 50;
	DenseMatrix X = swissroll(N);

	TapkeeOutput output;

	ASSERT_NO_THROW(output=tapkee::initialize()
			.withParameters((method=PCA, target_dimension=2))
			.embedUsing(X));

	auto projected = output.projection(X.row(0));
	ASSERT_EQ(2, projected.size());
}
