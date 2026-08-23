#include <gtest/gtest.h>

#include <tapkee/exceptions.hpp>
#include <tapkee/tapkee.hpp>

#include "callbacks.hpp"

using namespace tapkee;

TEST(Interface, ChainInterfaceOrder)
{
    const int N = 20;
    std::vector<float> indices(N);
    for (int i = 0; i < N; i++)
        indices[i] = i;

    float_kernel_callback kcb;
    float_distance_callback dcb;
    float_features_callback fcb;

    TapkeeOutput output;

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withKernel(kcb)
                                 .withFeatures(fcb)
                                 .withDistance(dcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withKernel(kcb)
                                 .withDistance(dcb)
                                 .withFeatures(fcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = KernelPrincipalComponentAnalysis))
                                 .withDistance(dcb)
                                 .withKernel(kcb)
                                 .withFeatures(fcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = PassThru))
                                 .withDistance(dcb)
                                 .withFeatures(fcb)
                                 .withKernel(kcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withFeatures(fcb)
                                 .withDistance(dcb)
                                 .withKernel(kcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withFeatures(fcb)
                                 .withKernel(kcb)
                                 .withDistance(dcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = PassThru))
                                 .withFeatures(fcb)
                                 .withKernel(kcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = PassThru))
                                 .withFeatures(fcb)
                                 .withDistance(dcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = KernelPrincipalComponentAnalysis))
                                 .withKernel(kcb)
                                 .withDistance(dcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = KernelPrincipalComponentAnalysis))
                                 .withKernel(kcb)
                                 .withFeatures(fcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withDistance(dcb)
                                 .withFeatures(fcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withDistance(dcb)
                                 .withKernel(kcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = KernelPrincipalComponentAnalysis))
                                 .withKernel(kcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = MultidimensionalScaling))
                                 .withDistance(dcb)
                                 .embedRange(indices.begin(), indices.end()));

    ASSERT_NO_THROW(output = tapkee::with((method = PassThru))
                                 .withFeatures(fcb)
                                 .embedRange(indices.begin(), indices.end()));
}

TEST(Interface, NoDataPassThru)
{
    std::vector<int> data;
    ASSERT_EQ(0, data.size());
    tapkee::dummy_kernel_callback<int> kcb;
    tapkee::dummy_distance_callback<int> dcb;
    tapkee::dummy_features_callback<int> fcb;

    TapkeeOutput output;
    // should produce no error
    ASSERT_THROW(output = tapkee::embed(data.begin(), data.end(), kcb, dcb, fcb, (method = PassThru)), no_data_error);
    // that's normal
    ASSERT_EQ(0, output.embedding.cols());
    // that's normal
    ASSERT_EQ(0, output.embedding.rows());
    // that's normal
    ASSERT_EQ(nullptr, output.projection.implementation.get());
}

TEST(Interface, ParameterTargetDimension)
{
    int td = 3;
    tapkee::Parameter target_dimension = tapkee::Parameter::create("td", td);
    ASSERT_EQ(td, static_cast<int>(target_dimension));
}

TEST(Interface, ParametersSet)
{
    int td = 3;
    int k = 5;
    tapkee::ParametersSet pg = tapkee::kwargs[(target_dimension = td, num_neighbors = k)];
    ASSERT_EQ(static_cast<int>(pg[target_dimension]), td);
    ASSERT_EQ(static_cast<int>(pg[num_neighbors]), k);
}

TEST(Interface, OneParameterParametersSet)
{
    int td = 3;
    tapkee::ParametersSet pg = tapkee::kwargs[(target_dimension = td)];
    ASSERT_EQ(static_cast<int>(pg[target_dimension]), td);
}

TEST(Interface, WrongParameterValueKernelLocallyLinearEmbedding)
{
    std::vector<float> data;
    data.push_back(0.0);
    ASSERT_EQ(1, data.size());
    float_kernel_callback kcb;
    tapkee::dummy_distance_callback<float> dcb;
    tapkee::dummy_features_callback<float> fcb;

    TapkeeOutput output;
    // fails with wrong parameter type as '-1' is not a valid value.
    ASSERT_THROW(output = embed(data.begin(), data.end(), kcb, dcb, fcb,
                                tapkee::kwargs[(method = KernelLocallyLinearEmbedding, num_neighbors = -3)]),
                 wrong_parameter_error);
}

TEST(Interface, WrongParameterTypeKernelLocallyLinearEmbedding)
{
    std::vector<float> data;
    for (int i = 0; i < 10; i++)
        data.push_back(i);
    float_kernel_callback kcb;
    tapkee::dummy_distance_callback<float> dcb;
    tapkee::dummy_features_callback<float> fcb;

    TapkeeOutput output;
    // fails as the number of neighbors is expected to be of IndexType, not float
    ASSERT_THROW(output = embed(data.begin(), data.end(), kcb, dcb, fcb,
                                tapkee::kwargs[(method = KernelLocallyLinearEmbedding,
                                                tapkee::Parameter::create("number of neighbors", 5.5f))]),
                 wrong_parameter_type_error);
}

TEST(Interface, ParameterWrongTypeCast)
{
    tapkee::Parameter p = tapkee::Parameter::create("p", 3);
    ASSERT_THROW(static_cast<double>(p), stichwort::wrong_parameter_type_error);
}

TEST(Interface, UninitializedParameterCast)
{
    tapkee::Parameter p;
    ASSERT_THROW(static_cast<int>(p), stichwort::missed_parameter_error);
}

struct NonStreamableValue
{
    int value;
};

TEST(Interface, ParameterRepresentation)
{
    tapkee::Parameter streamable = tapkee::Parameter::create("streamable", 3);
    ASSERT_EQ("3", streamable.repr());

    tapkee::Parameter non_streamable = tapkee::Parameter::create("non-streamable", NonStreamableValue{3});
    ASSERT_EQ("(can't obtain value)", non_streamable.repr());

    tapkee::Parameter uninitialized;
    ASSERT_EQ("uninitialized", uninitialized.repr());
}

TEST(Interface, ParameterWithDefault)
{
    tapkee::Parameter uninitialized;
    ASSERT_EQ(42, static_cast<int>(uninitialized.withDefault(42)));

    tapkee::Parameter initialized = tapkee::Parameter::create("initialized", 3);
    ASSERT_EQ(3, static_cast<int>(initialized.withDefault(42)));
}

TEST(Interface, ParameterEquality)
{
    tapkee::Parameter p = tapkee::Parameter::create("p", 5);
    ASSERT_TRUE(p.is(5));
    ASSERT_FALSE(p.is(6));
    // same value of a different type is not equal
    ASSERT_FALSE(p.is(5.0));
}

TEST(Interface, MultipleParameterKernelLocallyLinearEmbedding)
{
    std::vector<int> data;
    ASSERT_EQ(0, data.size());
    tapkee::dummy_kernel_callback<int> kcb;
    tapkee::dummy_distance_callback<int> dcb;
    tapkee::dummy_features_callback<int> fcb;

    tapkee::TapkeeOutput output;
    tapkee::ParametersSet parameters;
    ASSERT_THROW((output = tapkee::embed(
                      data.begin(), data.end(), kcb, dcb, fcb,
                      tapkee::kwargs[(method = KernelLocallyLinearEmbedding, num_neighbors = 6, num_neighbors = 5)])),
                 multiple_parameter_error);
}

bool always_cancel()
{
    return true;
}

TEST(Interface, CancellationPassThru)
{
    std::vector<int> data;
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    tapkee::dummy_kernel_callback<int> kcb;
    tapkee::dummy_distance_callback<int> dcb;
    tapkee::dummy_features_callback<int> fcb;

    TapkeeOutput output;
    // should cancel
    ASSERT_THROW(output = embed(data.begin(), data.end(), kcb, dcb, fcb,
                                tapkee::kwargs[(method = PassThru, cancel_function = always_cancel)]),
                 cancelled_exception);
}

TEST(Interface, NoReductionMethodSetFailPassThru)
{
    std::vector<int> data;
    ASSERT_EQ(0, data.size());
    tapkee::dummy_kernel_callback<int> kcb;
    tapkee::dummy_distance_callback<int> dcb;
    tapkee::dummy_features_callback<int> fcb;

    TapkeeOutput output;
    // should fail with missed parameter
    ASSERT_THROW(output = embed(data.begin(), data.end(), kcb, dcb, fcb, tapkee::kwargs[(eigen_method = Dense)]),
                 missed_parameter_error);
}

TEST(Interface, UnsupportedRandomizedForGeneralizedLE)
{
    std::vector<int> data;
    for (int i = 0; i < 20; i++)
        data.push_back(i);

    tapkee::dummy_kernel_callback<int> kcb;
    float_distance_callback dcb;
    tapkee::dummy_features_callback<int> fcb;

    TapkeeOutput output;
    ASSERT_THROW(output = embed(data.begin(), data.end(), kcb, dcb, fcb,
                                tapkee::kwargs[(method = LaplacianEigenmaps, eigen_method = Randomized)]),
                 unsupported_method_error);
}

TEST(Interface, EigenDecompositionFailMDS)
{
    const int N = 100;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++)
        data[i] = 0.0;

    tapkee::dummy_kernel_callback<float> kcb;
    float_distance_callback dcb;
    tapkee::dummy_features_callback<float> fcb;

    TapkeeOutput output;
    ASSERT_THROW(output = tapkee::embed(data.begin(), data.end(), kcb, dcb, fcb,
                                        tapkee::kwargs[(method = MultidimensionalScaling, eigen_method = Randomized)]),
                 eigendecomposition_error);
}

TEST(Interface, NotEnoughMemoryMDS)
{
    const int N = 10000000;
    std::vector<float> data(N);
    for (int i = 0; i < N; i++)
        data[i] = i;

    tapkee::dummy_kernel_callback<float> kcb;
    float_distance_callback dcb;
    tapkee::dummy_features_callback<float> fcb;

    tapkee::TapkeeOutput output;
    // tries to form 10000000 x 10000000 matrix (won't work on any machine in 2013)
    ASSERT_THROW(output = embed(data.begin(), data.end(), kcb, dcb, fcb,
                                tapkee::kwargs[(method = MultidimensionalScaling, eigen_method = Dense)]),
                 not_enough_memory_error);
}
