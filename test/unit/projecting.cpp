#include <gtest/gtest.h>

#include <tapkee/tapkee.hpp>

#include "data.hpp"

using namespace tapkee;

TEST(Projecting, PrincipalComponentAnalysis)
{
    const int N = 50;
    DenseMatrix X = swissroll(N);

    TapkeeOutput output;

    ASSERT_NO_THROW(output = tapkee::with((method = PrincipalComponentAnalysis, target_dimension = 2)).embedUsing(X));

    auto projected = output.projection(X.col(0));
    ASSERT_EQ(2, projected.size());
}

TEST(Projecting, RandomProjection)
{
    const int N = 50;
    DenseMatrix X = swissroll(N);

    TapkeeOutput output;

    ASSERT_NO_THROW(output = tapkee::with((method = RandomProjection, target_dimension = 2)).embedUsing(X));

    auto projected = output.projection(X.col(0));
    ASSERT_EQ(2, projected.size());
}
