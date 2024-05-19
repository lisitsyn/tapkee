/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */
#pragma once

#include <fstream>
#include <istream>
#include <iterator>
#include <ostream>

#include <tapkee/defines.hpp>
#include <tapkee/utils/logging.hpp>

using namespace std;

inline bool is_wrong_char(char c)
{
    if (!(isdigit(c) || isspace(c) || c == '.' || c == '-' || c == '+' || c == 'e'))
    {
        return true;
    }
    return false;
}

int levenshtein_distance(const std::string& s1, const std::string& s2)
{
    const auto len1 = s1.size();
    const auto len2 = s2.size();

    std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    d[0][0] = 0;
    for (unsigned int i = 1; i <= len1; ++i)
    {
        d[i][0] = i;
    }
    for (unsigned int j = 1; j <= len2; ++j)
    {
        d[0][j] = j;
    }

    for (unsigned int i = 1; i <= len1; ++i)
    {
        for (unsigned int j = 1; j <= len2; ++j)
            {
                d[i][j] = std::min({
                        d[i - 1][j] + 1,
                        d[i][j - 1] + 1,
                        d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1)
                });
            }
    }

    return d[len1][len2];
}

template <typename Iterator>
std::string comma_separated_keys(Iterator begin, Iterator end) {
    std::ostringstream oss;
    for (Iterator it = begin; it != end; ++it)
    {
        oss << it->first;
        if (std::next(it) != end)
        {
            oss << ", ";
        }
    }
    return oss.str();
}

tapkee::DenseMatrix read_data(ifstream& ifs, char delimiter)
{
    string str;
    vector<vector<tapkee::ScalarType>> input_data;
    while (ifs)
    {
        getline(ifs, str);

        // if (find_if(str.begin(), str.end(), is_wrong_char) != str.end())
        //	throw std::runtime_error("Input file contains some junk, please check it");

        istringstream ss(str);
        if (str.size())
        {
            vector<tapkee::ScalarType> row;
            while (ss)
            {
                string value_string;
                if (!getline(ss, value_string, delimiter))
                    break;
                istringstream value_stream(value_string);
                tapkee::ScalarType value;
                if (value_stream >> value)
                    row.push_back(value);
            }
            input_data.push_back(row);
        }
    }

    if (!input_data.empty())
    {
        tapkee::DenseMatrix fm(input_data.size(), input_data[0].size());
        for (int i = 0; i < fm.rows(); i++)
        {
            if (static_cast<tapkee::DenseMatrix::Index>(input_data[i].size()) != fm.cols())
            {
                stringstream ss;
                ss << "Wrong data at line " << i;
                throw std::runtime_error(ss.str());
            }
            for (int j = 0; j < fm.cols(); j++)
                fm(i, j) = input_data[i][j];
        }
        return fm;
    }
    else
    {
        return tapkee::DenseMatrix(0, 0);
    }
}

void write_matrix(tapkee::DenseMatrix* matrix, ofstream& of, char delimiter)
{
    for (int i = 0; i < matrix->rows(); i++)
    {
        for (int j = 0; j < matrix->cols(); j++)
        {
            of << (*matrix)(i, j);
            if (j != matrix->cols() - 1)
                of << delimiter;
        }
        of << endl;
    }
}

void write_vector(tapkee::DenseVector* matrix, ofstream& of)
{
    for (int i = 0; i < matrix->rows(); i++)
    {
        of << (*matrix)(i) << endl;
    }
}

static const std::map<std::string, tapkee::DimensionReductionMethod> DIMENSION_REDUCTION_METHODS = {
    {"local_tangent_space_alignment", tapkee::KernelLocalTangentSpaceAlignment},
    {"ltsa", tapkee::KernelLocalTangentSpaceAlignment},
    {"locally_linear_embedding", tapkee::KernelLocallyLinearEmbedding},
    {"lle", tapkee::KernelLocallyLinearEmbedding},
    {"hessian_locally_linear_embedding", tapkee::HessianLocallyLinearEmbedding},
    {"hlle", tapkee::HessianLocallyLinearEmbedding},
    {"multidimensional_scaling", tapkee::MultidimensionalScaling},
    {"mds", tapkee::MultidimensionalScaling},
    {"landmark_multidimensional_scaling", tapkee::LandmarkMultidimensionalScaling},
    {"l-mds", tapkee::LandmarkMultidimensionalScaling},
    {"isomap", tapkee::Isomap},
    {"landmark_isomap", tapkee::LandmarkIsomap},
    {"l-isomap", tapkee::LandmarkIsomap},
    {"diffusion_map", tapkee::DiffusionMap},
    {"dm", tapkee::DiffusionMap},
    {"kernel_pca", tapkee::KernelPrincipalComponentAnalysis},
    {"kpca", tapkee::KernelPrincipalComponentAnalysis},
    {"pca", tapkee::PrincipalComponentAnalysis},
    {"random_projection", tapkee::RandomProjection},
    {"ra", tapkee::RandomProjection},
    {"laplacian_eigenmaps", tapkee::LaplacianEigenmaps},
    {"la", tapkee::LaplacianEigenmaps},
    {"locality_preserving_projections", tapkee::LocalityPreservingProjections},
    {"lpp", tapkee::LocalityPreservingProjections},
    {"neighborhood_preserving_embedding", tapkee::NeighborhoodPreservingEmbedding},
    {"npe", tapkee::NeighborhoodPreservingEmbedding},
    {"linear_local_tangent_space_alignment", tapkee::LinearLocalTangentSpaceAlignment},
    {"lltsa", tapkee::LinearLocalTangentSpaceAlignment},
    {"stochastic_proximity_embedding", tapkee::StochasticProximityEmbedding},
    {"spe", tapkee::StochasticProximityEmbedding},
    {"passthru", tapkee::PassThru},
    {"factor_analysis", tapkee::FactorAnalysis},
    {"fa", tapkee::FactorAnalysis},
    {"t-stochastic_proximity_embedding", tapkee::tDistributedStochasticNeighborEmbedding},
    {"t-sne", tapkee::tDistributedStochasticNeighborEmbedding},
    {"manifold_sculpting", tapkee::ManifoldSculpting},
};

static const std::map<std::string, tapkee::NeighborsMethod> NEIGHBORS_METHODS = {
    {"brute", tapkee::Brute},
    {"vptree", tapkee::VpTree},
#ifdef TAPKEE_USE_LGPL_COVERTREE
    {"covertree", tapkee::CoverTree},
#endif
};

static const std::map<std::string, tapkee::EigenMethod> EIGEN_METHODS = {
    {"dense", tapkee::Dense},
    {"randomized", tapkee::Randomized},
#ifdef TAPKEE_WITH_ARPACK
    {"arpack", tapkee::Arpack},
#endif
};

static const std::map<std::string, tapkee::ComputationStrategy> COMPUTATION_STRATEGIES = {
    {"cpu", tapkee::HomogeneousCPUStrategy},
#ifdef TAPKEE_WITH_VIENNACL
    {"opencl", tapkee::HeterogeneousOpenCLStrategy},
#endif
};

template <class Mapping>
typename Mapping::mapped_type parse_multiple(Mapping mapping, const std::string& str)
{
    auto it = mapping.find(str);
    if (it != mapping.end())
    {
        return it->second;
    }

    auto closest = std::min_element(mapping.begin(), mapping.end(),
        [&str] (const auto &a, const auto &b) {
            return levenshtein_distance(str, a.first) < levenshtein_distance(str, b.first);
        });
    if (closest != mapping.end())
    {
        tapkee::Logging::instance().message_info(fmt::format("Unknown parameter value `{}`. Did you mean `{}`?", str, closest->first));
    }

    throw std::logic_error(str);
}

template <class PairwiseCallback>
tapkee::DenseMatrix matrix_from_callback(const tapkee::IndexType N, PairwiseCallback callback)
{
    tapkee::DenseMatrix result(N, N);
    tapkee::IndexType i, j;
#pragma omp parallel for shared(callback, result, N) private(j) default(none)
    for (i = 0; i < N; ++i)
    {
        for (j = i; j < N; j++)
        {
            tapkee::ScalarType res = callback(i, j);
            result(i, j) = res;
            result(j, i) = res;
        }
    }
    return result;
}
