#ifndef libedrt_defines
#define libedrt_defines

#include <vector>
#include <string>
#include <map>
#include <boost/any.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

typedef Eigen::Triplet<double> SparseTriplet;
typedef std::vector<SparseTriplet> SparseTriplets;
typedef Eigen::MatrixXd DenseMatrix;
typedef Eigen::VectorXd DenseVector;
typedef std::vector<int> LocalNeighbors;
typedef std::vector<LocalNeighbors> Neighbors;
typedef Eigen::SparseMatrix<double> WeightMatrix;
typedef Eigen::MatrixXd EmbeddingMatrix;
typedef std::map<std::string, boost::any> ParametersMap;

#endif
