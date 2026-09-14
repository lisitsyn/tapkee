#include <gtest/gtest.h>

#include <tapkee/callbacks/eigen_callbacks.hpp>
#include <tapkee/routines/manifold_sculpting.hpp>

using namespace tapkee;
using namespace tapkee::tapkee_internal;

TEST(ManifoldSculpting, SharedContinuationPointDoesNotCombineAngles)
{
    DenseMatrix data(2, 4);
    data << 0, 1, 1, 2,
            0, 0.1, -0.1, 0;
    const Neighbors neighbors{{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
    const std::vector<IndexType> indices{0, 1, 2, 3};
    ScalarType average_distance;
    const SparseMatrix distances = neighbors_distances_matrix(
        indices.begin(), indices.end(), neighbors, eigen_distance_callback(data), average_distance);
    const auto angles = angles_matrix_and_neighbors(neighbors, data);
    ASSERT_EQ(3, angles.second[0][0]);
    ASSERT_EQ(3, angles.second[0][1]);
    const std::set<IndexType> adjusted;
    const DataForErrorFunc error_data{distances, angles.first, neighbors, angles.second, adjusted, average_distance};
    // Both edges continue through point 3, but their angles are separate
    // constraints. The unmodified input must have zero geometric error.
    EXPECT_NEAR(0, compute_error_for_point(0, data, error_data), 1e-12);
}

TEST(ManifoldSculpting, CountOnlyPassesThatImproveThePoint)
{
    DenseMatrix data(2, 3);
    data << 0, 1, 2,
            0, 0, 0;
    const Neighbors neighbors{{1}, {0}, {1}};
    const std::vector<IndexType> indices{0, 1, 2};
    ScalarType average_distance;
    const SparseMatrix distances = neighbors_distances_matrix(
        indices.begin(), indices.end(), neighbors, eigen_distance_callback(data), average_distance);
    const auto angles = angles_matrix_and_neighbors(neighbors, data);
    const std::set<IndexType> adjusted;
    const DataForErrorFunc error_data{distances, angles.first, neighbors, angles.second, adjusted, average_distance};

    ScalarType point_error;
    // Already-optimal points must contribute zero to the adaptive step count.
    EXPECT_EQ(0, adjust_point_at_index(0, data, 2, 0.125, error_data, point_error));
    EXPECT_DOUBLE_EQ(0, point_error);

    data(1, 0) = 1;
    const ScalarType initial_error = compute_error_for_point(0, data, error_data);
    EXPECT_GT(adjust_point_at_index(0, data, 2, 0.125, error_data, point_error), 0);
    EXPECT_LT(point_error, initial_error);
    // Rechecking the local optimum must not claim an additional adjustment.
    EXPECT_EQ(0, adjust_point_at_index(0, data, 2, 0.125, error_data, point_error));
}
