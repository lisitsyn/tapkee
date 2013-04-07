/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * This code also uses Any type developed by C. Diggins under Boost license, version 1.0.
 * http://www.boost.org/LICENSE_1_0.txt
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_H_
#define TAPKEE_DEFINES_H_

/* Tapkee includes */
#include <tapkee_exceptions.hpp>
#include <utils/any.hpp>
#include <callback/callback_traits.hpp>
#include <routines/methods_traits.hpp>
/* End of Tapkee includes */

#include <map>
#include <vector>
#include <utility>

#define TAPKEE_MAJOR_VERSION 1
#define TAPKEE_MINOR_VERSION 0

//// Eigen 3 library includes
#ifdef TAPKEE_EIGEN_INCLUDE_FILE
	#include TAPKEE_EIGEN_INCLUDE_FILE
#else 
	#ifndef TAPKEE_DEBUG
		#define EIGEN_NO_DEBUG
	#endif
	#define EIGEN_RUNTIME_NO_MALLOC
	#include <Eigen/Eigen>
	#include <Eigen/Dense>
	#if EIGEN_VERSION_AT_LEAST(3,0,93)
		#include <Eigen/Sparse>
		#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
			#include <Eigen/SuperLUSupport>
		#endif
	#else
		#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		#include <unsupported/Eigen/SparseExtra>
	#endif
#endif

#ifdef EIGEN_RUNTIME_NO_MALLOC
	#define RESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(false);
	#define UNRESTRICT_ALLOC Eigen::internal::set_is_malloc_allowed(true);
#else
	#define RESTRICT_ALLOC
	#define UNRESTRICT_ALLOC
#endif
//// end of Eigen 3 library includes

//! Main namespace of the library, contains all public API definitions
namespace tapkee 
{

	//! Parameters that are used by the library
	enum ParameterKey
	{
		/** The key of the parameter map to indicate dimension reduction method that
		 * is going to be used.
		 *
		 * Should always be set in the parameter map.
		 * 
		 * The corresponding value should be of type @ref tapkee::TAPKEE_METHOD. 
		 */
		ReductionMethod,
		/** The key of the parameter map to store number of neighbors.
		 *
		 * Should be set for all local methods such as:
		 * 
		 * - @ref tapkee::KernelLocallyLinearEmbedding
		 * - @ref tapkee::NeighborhoodPreservingEmbedding
		 * - @ref tapkee::KernelLocalTangentSpaceAlignment
		 * - @ref tapkee::LinearLocalTangentSpaceAlignment
		 * - @ref tapkee::HessianLocallyLinearEmbedding
		 * - @ref tapkee::LaplacianEigenmaps
		 * - @ref tapkee::LocalityPreservingProjections
		 * - @ref tapkee::Isomap
		 * - @ref tapkee::LandmarkIsomap
		 * - @ref tapkee::StochasticProximityEmbedding (with local strategy, i.e. 
		 *        when @ref tapkee::SpeGlobalStrategy is set to false)
		 *
		 * The corresponding value should be of type @ref tapkee::IndexType, 
		 * greater than @ref MINIMAL_K (3) and less than 
		 * total number of vectors. 
		 */
		NumberOfNeighbors,
		/** The key of the parameter map to store target dimension.
		 *
		 * It should be set as it is used by all methods. By default it is 
		 * set to 2 though.
		 * 
		 * The corresponding value should have type 
		 * @ref tapkee::IndexType and be greater than 
		 * @ref MINIMAL_TD (1) and less than
		 * minimum of total number of vectors and 
		 * current dimension. 
		 */
		TargetDimension,
		/** The key of the parameter map to store current dimension.
		 *
		 * Should be set for the following methods:
		 *
		 *  - @ref tapkee::NeighborhoodPreservingEmbedding
		 *  - @ref tapkee::LinearLocalTangentSpaceAlignment
		 *  - @ref tapkee::LocalityPreservingProjections
		 *  - @ref tapkee::PCA
		 *  - @ref tapkee::RandomProjection
		 *  - @ref tapkee::PassThru
		 *  - @ref tapkee::FactorAnalysis
		 * 
		 * The corresponding value should have type @ref tapkee::IndexType and
		 * be greater than 1. 
		 */
		CurrentDimension,
		/** The key of the parameter map to indicate eigendecomposition
		 * method that is going to be used.
		 * 
		 * Should be set for the following eigendecomposition-based methods:
		 *
		 * - @ref tapkee::KernelLocallyLinearEmbedding
		 * - @ref tapkee::NeighborhoodPreservingEmbedding
		 * - @ref tapkee::KernelLocalTangentSpaceAlignment
		 * - @ref tapkee::LinearLocalTangentSpaceAlignment
		 * - @ref tapkee::HessianLocallyLinearEmbedding
		 * - @ref tapkee::LaplacianEigenmaps
		 * - @ref tapkee::LocalityPreservingProjections
		 * - @ref tapkee::DiffusionMap
		 * - @ref tapkee::Isomap
		 * - @ref tapkee::LandmarkIsomap
		 * - @ref tapkee::MultidimensionalScaling
		 * - @ref tapkee::LandmarkMultidimensionalScaling
		 * - @ref tapkee::KernelPCA
		 * - @ref tapkee::PCA
		 *
		 * By default it is set to @ref tapkee::ARPACK if available.
		 *
		 * The corresponding value should have type 
		 * @ref tapkee::EigenEmbeddingMethodId. 
		 */
		EigenEmbeddingMethod,
		/** The key of the parameter map to indicate neighbors
		 * finding method that is going to be used.
		 * 
		 * Should be set for the following local methods:
		 * 
		 * - @ref tapkee::KernelLocallyLinearEmbedding
		 * - @ref tapkee::NeighborhoodPreservingEmbedding
		 * - @ref tapkee::KernelLocalTangentSpaceAlignment
		 * - @ref tapkee::LinearLocalTangentSpaceAlignment
		 * - @ref tapkee::HessianLocallyLinearEmbedding
		 * - @ref tapkee::LaplacianEigenmaps
		 * - @ref tapkee::LocalityPreservingProjections
		 * - @ref tapkee::Isomap
		 * - @ref tapkee::LandmarkIsomap
		 * - @ref tapkee::StochasticProximityEmbedding (with local strategy, i.e. 
		 *        when @ref tapkee::SpeGlobalStrategy is set to false)
		 *
		 * By default it is set to @ref tapkee::COVERTREE if available.
		 *
		 * The corresponding value should have 
		 * type @ref tapkee::NeighborsMethodId.
		 */
		NeighborsMethod,
		/** The key of the parameter map to store number of
		 * 'timesteps' that should be made by diffusion map model.
		 * 
		 * Should be set for @ref tapkee::DiffusionMap only.
		 * 
		 * The corresponding value should have type @ref tapkee::IndexType.
		 */
		DiffusionMapTimesteps,
		/** The key of the parameter map to store width of
		 * the gaussian kernel, that is used by the 
		 * following methods:
		 *
		 * - @ref tapkee::LaplacianEigenmaps
		 * - @ref tapkee::LocalityPreservingProjections
		 * - @ref tapkee::DiffusionMap
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		GaussianKernelWidth,
		/** The key of the parameter map to store maximal 
		 * iteration that could be reached.
		 *
		 * Should be set for the following iterative methods:
		 * - @ref tapkee::StochasticProximityEmbedding
		 * - @ref tapkee::FactorAnalysis
		 * 
		 * The corresponding value should have type @ref tapkee::IndexType.
		 */
		MaxIteration,
		/** The key of the parameter map to indicate
		 * whether global strategy of SPE should be used.
		 *
		 * Should be set for @ref tapkee::StochasticProximityEmbedding only.
		 *
		 * The corresponding value should have type bool.
		 */
		SpeGlobalStrategy,
		/** The key of the parameter map to store number of
		 * updates to be done in SPE.
		 *
		 * Should be set for @ref tapkee::StochasticProximityEmbedding only.
		 *
		 * The corresponding value should have type @ref tapkee::IndexType.
		 */
		SpeNumberOfUpdates,
		/** The key of the parameter map to store tolerance of
		 * SPE. 
		 * 
		 * Should be set for @ref tapkee::StochasticProximityEmbedding only.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		SpeTolerance,
		/** The key of the parameter map to store ratio
		 * of landmark points to be used (1.0 means all
		 * points are landmarks and the reciprocal of number
		 * of vectors means only one landmark).
		 *
		 * Should be set for landmark methods:
		 *
		 * - @ref tapkee::LandmarkIsomap
		 * - @ref tapkee::LandmarkMultidimensionalScaling
		 *  
		 * The corresponding value should have type @ref tapkee::ScalarType
		 * and be in [0,1] range.
		 */
		LandmarkRatio,
		/** The key of the parameter map to store 
		 * diagonal shift regularizer coefficient
		 * of nullspace eigenproblems.
		 *
		 * Default is 1e-9.
		 *
		 * Should be set for the following methods:
		 *
		 * - @ref tapkee::KernelLocallyLinearEmbedding
		 * - @ref tapkee::NeighborhoodPreservingEmbedding
		 * - @ref tapkee::KernelLocalTangentSpaceAlignment
		 * - @ref tapkee::LinearLocalTangentSpaceAlignment
		 * - @ref tapkee::HessianLocallyLinearEmbedding
		 * - @ref tapkee::LocalityPreservingProjections
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType and
		 * be quite close to zero.
		 */
		NullspaceShift,
		/** The key of the parameter map to store
		 * regularization shift of locally linear embedding
		 * weights computation.
		 *
		 * Default is 1e-3.
		 *
		 * Should be set for @ref tapkee::KernelLocallyLinearEmbedding only.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType and
		 * be quite close to zero.
		 */
		KlleShift,
		/** The key of the parameter map to indicate
		 * whether graph connectivity check should be done.
		 *
		 * Default is true. 
		 *
		 * Should be set for the following local methods:
		 * 
		 * - @ref tapkee::KernelLocallyLinearEmbedding
		 * - @ref tapkee::NeighborhoodPreservingEmbedding
		 * - @ref tapkee::KernelLocalTangentSpaceAlignment
		 * - @ref tapkee::LinearLocalTangentSpaceAlignment
		 * - @ref tapkee::HessianLocallyLinearEmbedding
		 * - @ref tapkee::LaplacianEigenmaps
		 * - @ref tapkee::LocalityPreservingProjections
		 * - @ref tapkee::Isomap
		 * - @ref tapkee::LandmarkIsomap
		 * - @ref tapkee::StochasticProximityEmbedding (with local strategy, i.e. 
		 *        when @ref tapkee::SpeGlobalStrategy is set to false)
		 *
		 * The corresponding value should have type bool.
		 */
		CheckConnectivity,
		/** The key of the parameter map to store epsilon
		 * parameter of t
		 * 
		 * Should be set for @ref tapkee::FactorAnalysis only.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		FaEpsilon,
		/** The key of the parameter map to store a pointer
		 * to the function which could be called to indicate progress
		 * that was made (it is called with an argument in range [0,1],
		 * where 0 means 0% progress and 1 means 100% progress).
		 *
		 * Is not supported yet thus won't be used.
		 *
		 * The corresponding value should have type 
		 * @code void (*)(double) @endcode 
		 * (i.e. a pointer to some function that takes
		 *  double argument and returns nothing).
		 */
		ProgressFunction,
		/** The key of the parameter map to store a pointer 
		 * to the function which could be called to check if 
		 * computations were cancelled (the function should return 
		 * true if computations were cancelled).
		 *
		 * It is called only once when method is starting to work.
		 * 
		 * If function returns true the library immediately 
		 * throws @ref tapkee::cancelled_exception.
		 *
		 * The corresponding value should have type
		 * @code bool (*)() @endcode 
		 * (i.e. a pointer to some function that takes
		 *  nothing and returns boolean).
		 */
		CancelFunction,
		/** The key of the parameter map to store perplelixity
		 * parameter of t-SNE.
		 *
		 * Should be set for @ref tapkee::tDistributedStochasticNeighborEmbedding only.
		 *
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		SnePerplexity,
		/** The key of the parameter map to store theta 
		 * parameter of t-SNE.
		 *
		 * Should be set for @ref tapkee::tDistributedStochasticNeighborEmbedding only.
		 * 
		 * The corresponding value should have type @ref tapkee::ScalarType.
		 */
		SneTheta
	};


	//! Dimension reduction methods
	enum MethodId
	{
		/** Kernel Locally Linear Embedding as described in @cite Decoste2001 */
		KernelLocallyLinearEmbedding,
		/** Neighborhood Preserving Embedding as described in @cite He2005 */
		NeighborhoodPreservingEmbedding,
		/** Local Tangent Space Alignment as described in @cite Zhang2002 */
		KernelLocalTangentSpaceAlignment,
		/** Linear Local Tangent Space Alignment as described in @cite Zhang2007 */
		LinearLocalTangentSpaceAlignment,
		/** Hessian Locally Linear Embedding as described in @cite Donoho2003 */
		HessianLocallyLinearEmbedding,
		/** Laplacian Eigenmaps as described in @cite Belkin2002 */
		LaplacianEigenmaps,
		/** Locality Preserving Projections as described in @cite He2003 */
		LocalityPreservingProjections,
		/** Diffusion map as described in @cite Coifman2006 */
		DiffusionMap,
		/** Isomap as described in @cite Tenenbaum2000 */
		Isomap,
		/** Landmark Isomap as described in @cite deSilva2002 */
		LandmarkIsomap,
		/** Multidimensional scaling as described in @cite Cox2000 */
		MultidimensionalScaling,
		/** Landmark multidimensional scaling as described in @cite deSilva2004 */
		LandmarkMultidimensionalScaling,
		/** Stochastic Proximity Embedding as described in @cite Agrafiotis2003 */
		StochasticProximityEmbedding,
		/** Kernel PCA as described in @cite Scholkopf1997 */
		KernelPCA,
		/** Principal Component Analysis */
		PCA,
		/** Random Projection @cite Kaski1998*/
		RandomProjection,
		/** Factor Analysis */
		FactorAnalysis,
		/** t-SNE and Barnes-Hut-SNE as described in \cite tSNE and \cite Barnes-Hut-SNE */
		tDistributedStochasticNeighborEmbedding,
		/** Passing through (doing nothing just passes data through) */
		PassThru
	};


#ifndef DOXYGEN_SHOULD_SKIP_THIS
	// Methods identification
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KernelLocallyLinearEmbedding);
	METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(NeighborhoodPreservingEmbedding);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KernelLocalTangentSpaceAlignment);
	METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(LinearLocalTangentSpaceAlignment);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(HessianLocallyLinearEmbedding);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LaplacianEigenmaps);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(LocalityPreservingProjections);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(DiffusionMap);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(Isomap);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LandmarkIsomap);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(MultidimensionalScaling);
	METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(LandmarkMultidimensionalScaling);
	METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(StochasticProximityEmbedding);
	METHOD_THAT_NEEDS_ONLY_KERNEL_IS(KernelPCA);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(PCA);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(RandomProjection);
	METHOD_THAT_NEEDS_NOTHING_IS(PassThru);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(FactorAnalysis);
	METHOD_THAT_NEEDS_ONLY_FEATURES_IS(tDistributedStochasticNeighborEmbedding);
#endif // DOXYGEN_SHOULD_SKIP_THS


	//! Neighbors computation methods
	enum NeighborsMethodId
	{
		//! Brute force method with not least than 
		//! \f$ O(N N \log k) \f$ time complexity.
		//! Recommended to be used only in debug purposes.
		Brute,
#ifdef TAPKEE_USE_LGPL_COVERTREE
		//! Covertree-based method with approximate \f$ O(\log N) \f$ time complexity.
		//! Recommended to be used as a default method.
		CoverTree
#endif
	};


	//! Eigendecomposition methods
	enum EigenEmbeddingMethodId
	{
#ifdef TAPKEE_WITH_ARPACK
		//! ARPACK-based method (requires the ARPACK library
		//! binaries to be available around). Recommended to be used as a 
		//! default method. Supports both generalized and standard eigenproblems.
		Arpack,
#endif
		//! Randomized method (implementation taken from the redsvd lib). 
		//! Supports only standard but not generalized eigenproblems.
		Randomized,
		//! Eigen library dense method (could be useful for debugging). Computes
		//! all eigenvectors thus can be very slow doing large-scale.
		Dense
	};


#ifdef TAPKEE_CUSTOM_INTERNAL_NUMTYPE
	typedef TAPKEE_CUSTOM_INTERNAL_NUMTYPE ScalarType;
#else
	//! default scalar value (can be overrided with TAPKEE_CUSTOM_INTERNAL_NUMTYPE define)
	typedef double ScalarType;
#endif
	//! indexing type (non-overridable)
	//! set to int for compatibility with OpenMP 2.0
	typedef int IndexType;
	//! dense vector type (non-overridable)
	typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> DenseVector;
	//! dense matrix type (non-overridable) 
	typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,Eigen::Dynamic> DenseMatrix;
	//! dense symmetric matrix (non-overridable, currently just dense matrix, can be improved later)
	typedef DenseMatrix DenseSymmetricMatrix;
	//! sparse weight matrix type (non-overridable)
	typedef Eigen::SparseMatrix<ScalarType> SparseWeightMatrix;
	//! selfadjoint solver (non-overridable)
	typedef Eigen::SelfAdjointEigenSolver<DenseMatrix> DenseSelfAdjointEigenSolver;
	//! dense solver (non-overridable)
	typedef Eigen::LDLT<DenseMatrix> DenseSolver;
#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	//! sparse solver (it is Eigen::SimplicialCholesky in case of eigen version <3.1.0,
	//! in case of TAPKEE_USE_SUPERLU being defined it is Eigen::SuperLU, by default
	//! it is Eigen::SimplicialLDLT)
	typedef Eigen::SimplicialCholesky<SparseWeightMatrix> SparseSolver;
#else
	#if defined(TAPKEE_SUPERLU_AVAILABLE) && defined(TAPKEE_USE_SUPERLU)
		typedef Eigen::SuperLU<SparseWeightMatrix> SparseSolver;
	#else 
		typedef Eigen::SimplicialLDLT<SparseWeightMatrix> SparseSolver;
	#endif
#endif

#ifdef TAPKEE_CUSTOM_PROPERTIES
	#include TAPKEE_CUSTOM_PROPERTIES
#else
	//! Base of covertree. Could be overrided if TAPKEE_CUSTOM_PROPERTIES file is defined.
	#define COVERTREE_BASE 1.3
#endif

// Internal types (can be overrided)
#ifndef TAPKEE_INTERNAL_VECTOR
	#define TAPKEE_INTERNAL_VECTOR std::vector
#endif
#ifndef TAPKEE_INTERNAL_PAIR
	#define TAPKEE_INTERNAL_PAIR std::pair
#endif
#ifndef TAPKEE_INTERNAL_MAP
	#define TAPKEE_INTERNAL_MAP std::map
#endif
	
//! Parameters map with keys being values of @ref TAPKEE_PARAMETERS and 
//! values set to corresponding values wrapped to @ref any type
typedef TAPKEE_INTERNAL_MAP<ParameterKey, any> ParametersMap;

namespace tapkee_internal 
{
#if defined(TAPKEE_USE_PRIORITY_QUEUE) && defined(TAPKEE_USE_FIBONACCI_HEAP)
	#error "Can't use both priority queue and fibonacci heap at the same time"
#endif
#if !defined(TAPKEE_USE_PRIORITY_QUEUE) && !defined(TAPKEE_USE_FIBONACCI_HEAP)
	#define TAPKEE_USE_PRIORITY_QUEUE
#endif

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	template <typename T> struct Triplet
	{
		Triplet(IndexType colIndex, IndexType rowIndex, T valueT) : 
			col_(colIndex), row_(rowIndex), value_(valueT)
		{
		}
		IndexType col() const { return col_; };
		IndexType row() const { return row_; };
		T value() const { return value_; };
		IndexType col_;
		IndexType row_;
		T value_;
	};
	typedef Triplet<ScalarType> SparseTriplet;
#else // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	typedef Eigen::Triplet<ScalarType> SparseTriplet;
#endif // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
} // End of namespace tapkee_internal

namespace tapkee_internal
{
	typedef TAPKEE_INTERNAL_VECTOR<SparseTriplet> SparseTriplets;
	typedef TAPKEE_INTERNAL_VECTOR<IndexType> LocalNeighbors;
	typedef TAPKEE_INTERNAL_VECTOR<LocalNeighbors> Neighbors;
	typedef TAPKEE_INTERNAL_PAIR<DenseMatrix,DenseVector> EmbeddingResult;
	typedef Eigen::DiagonalMatrix<ScalarType,Eigen::Dynamic> DenseDiagonalMatrix;
	typedef TAPKEE_INTERNAL_VECTOR<IndexType> Landmarks;
	typedef TAPKEE_INTERNAL_PAIR<SparseWeightMatrix,DenseDiagonalMatrix> Laplacian;
	typedef TAPKEE_INTERNAL_PAIR<DenseSymmetricMatrix,DenseSymmetricMatrix> DenseSymmetricMatrixPair;

} // End of namespace tapkee_internal
} // End of namespace tapkee

/* Tapkee includes */
#include <tapkee_projection.hpp>
#include <utils/naming.hpp>
/* End of Tapkee includes */

namespace tapkee 
{

	//! Return result of the library - a pair of @ref DenseMatrix (embedding) and @ref ProjectingFunction
	typedef TAPKEE_INTERNAL_PAIR<DenseMatrix,tapkee::ProjectingFunction> ReturnResult;

} // End of namespace tapkee

#endif // TAPKEE_DEFINES_H_
