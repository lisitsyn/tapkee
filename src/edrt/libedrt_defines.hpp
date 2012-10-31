#ifndef libedrt_defines
#define libedrt_defines

#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

typedef Eigen::MatrixXd DenseMatrix;
typedef std::vector<int> LocalNeighbors;
typedef std::vector<LocalNeighbors> Neighbors;
//typedef Eigen::MatrixXd WeightMatrix;
typedef Eigen::SparseMatrix<double> WeightMatrix;
typedef Eigen::MatrixXd EmbeddingMatrix;

#endif
