/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines/types.hpp>
/* End of Tapkee includes */

namespace tapkee
{

template <typename M> struct Method
{
    Method(const char* n) : name_(n)
    {
    }
    Method(const M& m) : name_(m.name_)
    {
    }
    const char* name() const
    {
        return name_;
    }
    bool is(const M& m) const
    {
        return this->name() == m.name();
    }
    bool operator==(const M& m) const
    {
        return this->name() == m.name();
    }
    const char* name_;
};

template <typename M>
std::ostream& operator<<(std::ostream& os, const Method<M>& method)
{
    os << method.name();
    return os;
}

struct DimensionReductionTraits
{
    const bool needs_kernel;
    const bool needs_distance;
    const bool needs_features;
};

struct DimensionReductionMethod : public Method<DimensionReductionMethod>
{
    DimensionReductionMethod(const char* n, const DimensionReductionTraits& traits)
        : Method<DimensionReductionMethod>(n)
        , needs_kernel(traits.needs_kernel)
        , needs_distance(traits.needs_distance)
        , needs_features(traits.needs_features)
    {
    }
    using Method<DimensionReductionMethod>::operator==;
    bool needs_kernel;
    bool needs_distance;
    bool needs_features;
};

static const DimensionReductionTraits RequiresKernel{true, false, false};
static const DimensionReductionTraits RequiresKernelAndFeatures{true, false, true};
static const DimensionReductionTraits RequiresDistance{false, true, false};
static const DimensionReductionTraits RequiresDistanceAndFeatures{false, true, true};
static const DimensionReductionTraits RequiresFeatures{false, false, true};

/** Kernel Locally Linear Embedding as described
 * in @cite Decoste2001 */
static const DimensionReductionMethod KernelLocallyLinearEmbedding("Kernel Locally Linear Embedding (KLLE)", RequiresKernel);

/** Neighborhood Preserving Embedding as described
 * in @cite He2005 */
static const DimensionReductionMethod NeighborhoodPreservingEmbedding("Neighborhood Preserving Embedding (NPE)", RequiresKernelAndFeatures);

/** Local Tangent Space Alignment as described
 * in @cite Zhang2002 */
static const DimensionReductionMethod KernelLocalTangentSpaceAlignment("Kernel Local Tangent Space Alignment (KLTSA)", RequiresKernel);

/** Linear Local Tangent Space Alignment as described
 * in @cite Zhang2007 */
static const DimensionReductionMethod LinearLocalTangentSpaceAlignment("Linear Local Tangent Space Alignment (LLTSA)", RequiresKernelAndFeatures);

/** Hessian Locally Linear Embedding as described in
 * @cite Donoho2003 */
static const DimensionReductionMethod HessianLocallyLinearEmbedding("Hessian Locally Linear Embedding (HLLE)", RequiresKernel);

/** Laplacian Eigenmaps as described in
 * @cite Belkin2002 */
static const DimensionReductionMethod LaplacianEigenmaps("Laplacian Eigenmaps", RequiresDistance);

/** Locality Preserving Projections as described in
 * @cite He2003 */
static const DimensionReductionMethod LocalityPreservingProjections("Locality Preserving Projections (LPP)", RequiresDistanceAndFeatures);

/** Diffusion map as described in
 * @cite Coifman2006 */
static const DimensionReductionMethod DiffusionMap("Diffusion Map", RequiresDistance);

/** Isomap as described in
 * @cite Tenenbaum2000 */
static const DimensionReductionMethod Isomap("Isomap", RequiresDistance);

/** Landmark Isomap as described in
 * @cite deSilva2002 */
static const DimensionReductionMethod LandmarkIsomap("Landmark Isomap", RequiresDistance);

/** Multidimensional scaling as described in
 * @cite Cox2000 */
static const DimensionReductionMethod MultidimensionalScaling("Multidimensional Scaling (MDS)", RequiresDistance);

/** Landmark multidimensional scaling as described in
 * @cite deSilva2004 */
static const DimensionReductionMethod LandmarkMultidimensionalScaling("Landmark Multidimensional Scaling (LMDS)", RequiresDistance);

/** Stochastic Proximity Embedding as described in
 * @cite Agrafiotis2003 */
static const DimensionReductionMethod StochasticProximityEmbedding("Stochastic Proximity Embedding (SPE)", RequiresDistanceAndFeatures);

/** Kernel PCA as described in
 * @cite Scholkopf1997 */
static const DimensionReductionMethod KernelPrincipalComponentAnalysis("Kernel Principal Component Analysis (KPCA)", RequiresKernel);

/** Principal Component Analysis */
static const DimensionReductionMethod PrincipalComponentAnalysis("Principal Component Analysis (PCA)", RequiresFeatures);

/** Random Projection as described in
 * @cite Kaski1998*/
static const DimensionReductionMethod RandomProjection("Random Projection", RequiresFeatures);

/** Factor Analysis */
static const DimensionReductionMethod FactorAnalysis("Factor Analysis", RequiresFeatures);

/** t-SNE and Barnes-Hut-SNE as described in
 * @cite vanDerMaaten2008 and @cite vanDerMaaten2013 */
static const DimensionReductionMethod tDistributedStochasticNeighborEmbedding("t-distributed Stochastic Neighbor Embedding (t-SNE)", RequiresFeatures);

/** Manifold Sculpting as described in
 * @cite Gashler2007 */
static const DimensionReductionMethod ManifoldSculpting("Manifold Sculpting", RequiresFeatures);

/** Passing through (doing nothing just passes the
 * data through) */
static const DimensionReductionMethod PassThru("Pass-through", RequiresFeatures);

struct NeighborsMethod : public Method<NeighborsMethod>
{
    NeighborsMethod(const char* n) : Method<NeighborsMethod>(n)
    {
    }
    using Method<NeighborsMethod>::operator==;
};

//! Brute force method with not least than
//! \f$ O(N N \log k) \f$ time complexity.
//! Recommended to be used only in debug purposes.
static const NeighborsMethod Brute("Brute-force");
//! Vantage point tree -based method.
static const NeighborsMethod VpTree("Vantage point tree");
#ifdef TAPKEE_USE_LGPL_COVERTREE
//! Covertree-based method with approximate \f$ O(\log N) \f$ time complexity.
//! Recommended to be used as a default method.
static const NeighborsMethod CoverTree("Cover tree");
#endif

#ifdef TAPKEE_USE_LGPL_COVERTREE
static NeighborsMethod default_neighbors_method = CoverTree;
#else
static NeighborsMethod default_neighbors_method = Brute;
#endif

struct EigenMethod : public Method<EigenMethod>
{
    EigenMethod(const char* n) : Method<EigenMethod>(n)
    {
    }
    using Method<EigenMethod>::operator==;
};

#ifdef TAPKEE_WITH_ARPACK
//! ARPACK-based method (requires the ARPACK library
//! binaries to be available around). Recommended to be used as a
//! default method. Supports both generalized and standard eigenproblems.
static const EigenMethod Arpack("Arpack");
#endif
//! Randomized method (implementation taken from the redsvd lib).
//! Supports only standard but not generalized eigenproblems.
static const EigenMethod Randomized("Randomized");
//! Eigen library dense method (could be useful for debugging). Computes
//! all eigenvectors thus can be very slow doing large-scale.
static const EigenMethod Dense("Dense");

#ifdef TAPKEE_WITH_ARPACK
static EigenMethod default_eigen_method = Arpack;
#else
static EigenMethod default_eigen_method = Dense;
#endif

struct ComputationStrategy : public Method<ComputationStrategy>
{
    ComputationStrategy(const char* n) : Method<ComputationStrategy>(n)
    {
    }
    using Method<ComputationStrategy>::operator==;
};

#ifdef TAPKEE_WITH_VIENNACL
static const ComputationStrategy HeterogeneousOpenCLStrategy("OpenCL");
#endif
static const ComputationStrategy HomogeneousCPUStrategy("CPU");

static ComputationStrategy default_computation_strategy = HomogeneousCPUStrategy;

namespace tapkee_internal
{

struct EigendecompositionStrategy : public Method<EigendecompositionStrategy>
{
    EigendecompositionStrategy(const char* n, IndexType skp) : Method<EigendecompositionStrategy>(n), skip_(skp)
    {
    }
    IndexType skip() const
    {
        return skip_;
    }
    const IndexType skip_;
};

static const EigendecompositionStrategy LargestEigenvalues("Largest eigenvalues", 0);
static const EigendecompositionStrategy SquaredLargestEigenvalues("Largest eigenvalues of squared matrix", 0);
static const EigendecompositionStrategy SmallestEigenvalues("Smallest eigenvalues", 1);

} // namespace tapkee_internal

} // namespace tapkee
