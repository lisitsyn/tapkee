/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

/* Tapkee includes */
#include <tapkee_defines.hpp>
#include <utils/time.hpp>
#include <utils/logging.hpp>
#include <utils/parameters.hpp>
#include <routines/locally_linear.hpp>
#include <routines/eigen_embedding.hpp>
#include <routines/generalized_eigen_embedding.hpp>
#include <routines/multidimensional_scaling.hpp>
#include <routines/diffusion_maps.hpp>
#include <routines/laplacian_eigenmaps.hpp>
#include <routines/isomap.hpp>
#include <routines/pca.hpp>
#include <routines/random_projection.hpp>
#include <routines/spe.hpp>
#include <routines/fa.hpp>
#include <neighbors/neighbors.hpp>
#include <external/barnes_hut_sne/tsne.hpp>
/* End of Tapkee includes */

using std::string;

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{

string get_method_name(TAPKEE_METHOD m)
{
	switch (m)
	{
		case KERNEL_LOCALLY_LINEAR_EMBEDDING: return "Kernel Locally Linear Embedding";
		case KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT: return "Local Tangent Space Alignment";
		case DIFFUSION_MAP: return "Diffusion Map";
		case MULTIDIMENSIONAL_SCALING: return "Classic Multidimensional Scaling";
		case LANDMARK_MULTIDIMENSIONAL_SCALING: return "Landmark Multidimensional Scaling";
		case ISOMAP: return "Isomap";
		case LANDMARK_ISOMAP: return "Landmark Isomap";
		case NEIGHBORHOOD_PRESERVING_EMBEDDING: return "Neighborhood Preserving Embedding";
		case LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT: return "Linear Local Tangent Space Alignment";
		case HESSIAN_LOCALLY_LINEAR_EMBEDDING: return "Hessian Locally Linear Embedding";
		case LAPLACIAN_EIGENMAPS: return "Laplacian Eigenmaps";
		case LOCALITY_PRESERVING_PROJECTIONS: return "Locality Preserving Embedding";
		case PCA: return "Principal Component Analysis";
		case KERNEL_PCA: return "Kernel Principal Component Analysis";
		case STOCHASTIC_PROXIMITY_EMBEDDING: return "Stochastic Proximity Embedding";
		case PASS_THRU: return "passing through";
		case RANDOM_PROJECTION: return "Random Projection";
		case FACTOR_ANALYSIS: return "Factor Analysis";
		case T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING: return "t-distributed Stochastic Neighbor Embedding";
		case UNKNOWN_METHOD: return "this should not happen, call the police";
	}
	return "hello";
}

class Context
{
public:
	Context(void (*progress)(double), bool (*cancel)()) :
		progress_function(progress), cancel_function(cancel)
	{
	}

	inline void report_progress(double x) const
	{
		if (progress_function)
			progress_function(x);
	}

	inline bool is_cancelled() const
	{
		if (cancel_function)
			return cancel_function();
		return false;
	}

private:
	void (*progress_function)(double);
	bool (*cancel_function)();
};

template <class KernelCallback, class DistanceCallback, class FeatureVectorCallback>
class Callbacks
{
public:
	Callbacks(KernelCallback kernel_callback, DistanceCallback distance_callback, 
	          FeatureVectorCallback feature_callback) : 
		kernel(kernel_callback), distance(distance_callback), feature(feature_callback) 
	{
	}

	KernelCallback kernel;
	DistanceCallback distance;
	FeatureVectorCallback feature;
};

#define IMPLEMENTATION_OF(METHOD)                                                     \
template <class RandomAccessIterator, class KernelCallback,                           \
          class DistanceCallback, class FeatureVectorCallback>                        \
ReturnResult METHOD##_implementation(                                                 \
   RandomAccessIterator begin, RandomAccessIterator end,                              \
   const Callbacks<KernelCallback,DistanceCallback,FeatureVectorCallback>& callbacks, \
   ParametersMap& parameters, const Context& context)

// eigenvalues parameters
#define SKIP_ONE_EIGENVALUE 1
#define SKIP_NO_EIGENVALUES 0

// minimal values
#define MINIMAL_K static_cast<IndexType>(3)
#define MINIMAL_TD static_cast<IndexType>(1)

// other useful macro
#define NUM_VECTORS static_cast<IndexType>(end-begin)
#define DO_MEASURE_RUN(X) timed_context timing_context__("[+] Embedding with " X)
#define STOP_IF_CANCELLED if (context.is_cancelled()) throw cancelled_exception()

IMPLEMENTATION_OF(KERNEL_LOCALLY_LINEAR_EMBEDDING)
{
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(ScalarType,                    eigenshift,         EIGENSHIFT);
	PARAMETER(ScalarType,                    traceshift,         KLLE_TRACE_SHIFT);
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("KLLE");
	STOP_IF_CANCELLED;

	Neighbors neighbors =
		find_neighbors(neighbors_method,begin,end,callbacks.kernel,k,check_connectivity);
	SparseWeightMatrix weight_matrix =
		linear_weight_matrix(begin,end,neighbors,callbacks.kernel,eigenshift,traceshift);
	return ReturnResult(eigen_embedding<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
		weight_matrix,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(KERNEL_LOCAL_TANGENT_SPACE_ALIGNMENT)
{
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(ScalarType,                    eigenshift,         EIGENSHIFT);
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("KLTSA");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.kernel,k,check_connectivity);
	SparseWeightMatrix weight_matrix = 
		tangent_weight_matrix(begin,end,neighbors,callbacks.kernel,target_dimension,eigenshift);
	return ReturnResult(eigen_embedding<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
		weight_matrix,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(DIFFUSION_MAP)
{
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,     EIGEN_EMBEDDING_METHOD,  NOT(eigen_method, UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     target_dimension, TARGET_DIMENSION,        IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(ScalarType,                    width,            GAUSSIAN_KERNEL_WIDTH,   POSITIVE(width));
	PARAMETER(IndexType,                     timesteps,        DIFFUSION_MAP_TIMESTEPS);
	
	DO_MEASURE_RUN("diffusion map");
	STOP_IF_CANCELLED;

	DenseSymmetricMatrix diffusion_matrix =
		compute_diffusion_matrix(begin,end,callbacks.distance,timesteps,width);
	return ReturnResult(eigen_embedding<DenseSymmetricMatrix,
			#ifdef TAPKEE_GPU
				GPUDenseImplicitSquareMatrixOperation
			#else 
				DenseImplicitSquareSymmetricMatrixOperation 
			#endif
			>(eigen_method,
		diffusion_matrix,target_dimension,SKIP_NO_EIGENVALUES).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(MULTIDIMENSIONAL_SCALING)
{
	PARAMETER(IndexType,                     target_dimension, TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,     EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));

	DO_MEASURE_RUN("MDS");
	STOP_IF_CANCELLED;

	DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,callbacks.distance);
	centerMatrix(distance_matrix);
	distance_matrix.array() *= -0.5;
	EmbeddingResult result = eigen_embedding<DenseSymmetricMatrix,
			#ifdef TAPKEE_GPU
					GPUDenseMatrixOperation
			#else
					DenseMatrixOperation
			#endif
			>(eigen_method,
		distance_matrix,target_dimension,SKIP_NO_EIGENVALUES);

	for (IndexType i=0; i<target_dimension; i++)
		result.first.col(i).array() *= sqrt(result.second(i));
	return ReturnResult(result.first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(LANDMARK_MULTIDIMENSIONAL_SCALING)
{
	PARAMETER(IndexType,                     target_dimension, TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,     EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(ScalarType,                    ratio,            LANDMARK_RATIO,         IN_RANGE(ratio,1/(NUM_VECTORS),1.0));

	DO_MEASURE_RUN("Landmark MDS");
	STOP_IF_CANCELLED;

	Landmarks landmarks = 
		select_landmarks_random(begin,end,ratio);
	DenseSymmetricMatrix distance_matrix = 
		compute_distance_matrix(begin,end,landmarks,callbacks.distance);
	DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
	centerMatrix(distance_matrix);
	distance_matrix.array() *= -0.5;
	EmbeddingResult landmarks_embedding = 
		eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
				distance_matrix,target_dimension,SKIP_NO_EIGENVALUES);
	for (IndexType i=0; i<target_dimension; i++)
		landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
	return ReturnResult(triangulate(begin,end,callbacks.distance,landmarks,
		landmark_distances_squared,landmarks_embedding,target_dimension).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(ISOMAP)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("Isomap");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.distance,k,check_connectivity);
	DenseSymmetricMatrix shortest_distances_matrix = 
		compute_shortest_distances_matrix(begin,end,neighbors,callbacks.distance);
	shortest_distances_matrix = shortest_distances_matrix.array().square();
	centerMatrix(shortest_distances_matrix);
	shortest_distances_matrix.array() *= -0.5;
	
	EmbeddingResult embedding = eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
		shortest_distances_matrix,target_dimension,SKIP_NO_EIGENVALUES);

	for (IndexType i=0; i<target_dimension; i++)
		embedding.first.col(i).array() *= sqrt(embedding.second(i));
	
	return ReturnResult(embedding.first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(LANDMARK_ISOMAP)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(ScalarType,                    ratio,              LANDMARK_RATIO,         IN_RANGE(ratio,1/(NUM_VECTORS),1.0));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("Landmark Isomap");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.distance,k,check_connectivity);
	Landmarks landmarks = 
		select_landmarks_random(begin,end,ratio);
	DenseMatrix distance_matrix = 
		compute_shortest_distances_matrix(begin,end,landmarks,neighbors,callbacks.distance);
	distance_matrix = distance_matrix.array().square();
	
	DenseVector col_means = distance_matrix.colwise().mean();
	DenseVector row_means = distance_matrix.rowwise().mean();
	ScalarType grand_mean = distance_matrix.mean();
	distance_matrix.array() += grand_mean;
	distance_matrix.colwise() -= row_means;
	distance_matrix.rowwise() -= col_means.transpose();
	distance_matrix.array() *= -0.5;

	EmbeddingResult landmarks_embedding;
	
	if (eigen_method==EIGEN_DENSE_SELFADJOINT_SOLVER)
	{
		DenseMatrix distance_matrix_sym = distance_matrix*distance_matrix.transpose();
		landmarks_embedding = eigen_embedding<DenseSymmetricMatrix,DenseImplicitSquareMatrixOperation>
			(eigen_method,distance_matrix_sym,target_dimension,SKIP_NO_EIGENVALUES);
	}
	else 
	{
		landmarks_embedding = eigen_embedding<DenseSymmetricMatrix,DenseImplicitSquareMatrixOperation>
			(eigen_method,distance_matrix,target_dimension,SKIP_NO_EIGENVALUES);
	}

	DenseMatrix embedding = distance_matrix.transpose()*landmarks_embedding.first;

	for (IndexType i=0; i<target_dimension; i++)
		embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
	return ReturnResult(embedding,tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(NEIGHBORHOOD_PRESERVING_EMBEDDING)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(IndexType,                     dimension,          CURRENT_DIMENSION,      POSITIVE(dimension));
	PARAMETER(ScalarType,                    eigenshift,         EIGENSHIFT);
	PARAMETER(ScalarType,                    traceshift,         KLLE_TRACE_SHIFT);
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("NPE");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.kernel,k,check_connectivity);
	SparseWeightMatrix weight_matrix = 
		linear_weight_matrix(begin,end,neighbors,callbacks.kernel,eigenshift,traceshift);
	DenseSymmetricMatrixPair eig_matrices =
		construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
			callbacks.feature,dimension);
	EmbeddingResult projection_result = 
		generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
			eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SKIP_NO_EIGENVALUES);
	DenseVector mean_vector = 
		compute_mean(begin,end,callbacks.feature,dimension);
	tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
	return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,dimension),projecting_function);
}

IMPLEMENTATION_OF(HESSIAN_LOCALLY_LINEAR_EMBEDDING)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("HLLE");
	STOP_IF_CANCELLED;

	Neighbors neighbors =
		find_neighbors(neighbors_method,begin,end,callbacks.kernel,k,check_connectivity);
	SparseWeightMatrix weight_matrix =
		hessian_weight_matrix(begin,end,neighbors,callbacks.kernel,target_dimension);
	return ReturnResult(eigen_embedding<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
		weight_matrix,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(LAPLACIAN_EIGENMAPS)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(ScalarType,                    width,              GAUSSIAN_KERNEL_WIDTH,  POSITIVE(width));
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("Laplacian Eigenmaps");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.distance,k,check_connectivity);
	Laplacian laplacian = 
		compute_laplacian(begin,end,neighbors,callbacks.distance,width);
	return ReturnResult(generalized_eigen_embedding<SparseWeightMatrix,DenseDiagonalMatrix,SparseInverseMatrixOperation>(
		eigen_method,laplacian.first,laplacian.second,target_dimension,SKIP_ONE_EIGENVALUE).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(LOCALITY_PRESERVING_PROJECTIONS)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(ScalarType,                    width,              GAUSSIAN_KERNEL_WIDTH,  POSITIVE(width));
	PARAMETER(IndexType,                     dimension,          CURRENT_DIMENSION,      POSITIVE(dimension));
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("LPP");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.distance,k,check_connectivity);
	Laplacian laplacian = 
		compute_laplacian(begin,end,neighbors,callbacks.distance,width);
	DenseSymmetricMatrixPair eigenproblem_matrices =
		construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
				callbacks.feature,dimension);
	EmbeddingResult projection_result = 
		generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
			eigen_method,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension,SKIP_NO_EIGENVALUES);
	DenseVector mean_vector = 
		compute_mean(begin,end,callbacks.feature,dimension);
	tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
	return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,dimension),projecting_function);
}

IMPLEMENTATION_OF(PCA)
{
	PARAMETER(IndexType,                     target_dimension, TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,     EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     dimension,        CURRENT_DIMENSION,      POSITIVE(dimension));

	DO_MEASURE_RUN("PCA");
	STOP_IF_CANCELLED;

	DenseVector mean_vector = 
		compute_mean(begin,end,callbacks.feature,dimension);
	DenseSymmetricMatrix centered_covariance_matrix = 
		compute_covariance_matrix(begin,end,mean_vector,callbacks.feature,dimension);
	EmbeddingResult projection_result = 
		eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_covariance_matrix,target_dimension,SKIP_NO_EIGENVALUES);
	tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
	return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,dimension), projecting_function);
}

IMPLEMENTATION_OF(RANDOM_PROJECTION)
{
	PARAMETER(IndexType, target_dimension, TARGET_DIMENSION,  IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(IndexType, dimension,        CURRENT_DIMENSION, POSITIVE(dimension));

	DO_MEASURE_RUN("Random Projection");
	STOP_IF_CANCELLED;

	DenseMatrix projection_matrix = 
		gaussian_projection_matrix(dimension, target_dimension);

	DenseVector mean_vector = 
		compute_mean(begin,end,callbacks.feature,dimension);

	tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_matrix,mean_vector));

	return ReturnResult(project(projection_matrix,mean_vector,begin,end,callbacks.feature,dimension), projecting_function);
}

IMPLEMENTATION_OF(KERNEL_PCA)
{
	PARAMETER(IndexType,                     target_dimension, TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,     EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));

	DO_MEASURE_RUN("kPCA");
	STOP_IF_CANCELLED;

	DenseSymmetricMatrix centered_kernel_matrix = 
		compute_centered_kernel_matrix(begin,end,callbacks.kernel);
	return ReturnResult(eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
		centered_kernel_matrix,target_dimension,SKIP_NO_EIGENVALUES).first, tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(LINEAR_LOCAL_TANGENT_SPACE_ALIGNMENT)
{
	PARAMETER(IndexType,                     target_dimension,   TARGET_DIMENSION,       IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(TAPKEE_EIGEN_EMBEDDING_METHOD, eigen_method,       EIGEN_EMBEDDING_METHOD, NOT(eigen_method,UNKNOWN_EIGEN_METHOD));
	PARAMETER(IndexType,                     k,                  NUMBER_OF_NEIGHBORS,    IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD,       neighbors_method,   NEIGHBORS_METHOD,       NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(IndexType,                     dimension,          CURRENT_DIMENSION,      POSITIVE(dimension));
	PARAMETER(ScalarType,                    eigenshift,         EIGENSHIFT);
	PARAMETER(bool,                          check_connectivity, CHECK_CONNECTIVITY);

	DO_MEASURE_RUN("LLTSA");
	STOP_IF_CANCELLED;

	Neighbors neighbors = 
		find_neighbors(neighbors_method,begin,end,callbacks.kernel,k,check_connectivity);
	SparseWeightMatrix weight_matrix = 
		tangent_weight_matrix(begin,end,neighbors,callbacks.kernel,target_dimension,eigenshift);
	DenseSymmetricMatrixPair eig_matrices =
		construct_lltsa_eigenproblem(weight_matrix,begin,end,
			callbacks.feature,dimension);
	EmbeddingResult projection_result = 
		generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
			eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SKIP_NO_EIGENVALUES);
	DenseVector mean_vector = 
		compute_mean(begin,end,callbacks.feature,dimension);
	tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
	return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,dimension),
			projecting_function);
}

IMPLEMENTATION_OF(STOCHASTIC_PROXIMITY_EMBEDDING)
{
	PARAMETER(IndexType,               target_dimension,   TARGET_DIMENSION,    IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(IndexType,               k,                  NUMBER_OF_NEIGHBORS, IN_RANGE(k,MINIMAL_K,NUM_VECTORS));
	PARAMETER(TAPKEE_NEIGHBORS_METHOD, neighbors_method,   NEIGHBORS_METHOD,    NOT(neighbors_method,UNKNOWN_NEIGHBORS_METHOD));
	PARAMETER(ScalarType,              tolerance,          SPE_TOLERANCE,       POSITIVE(tolerance));
	PARAMETER(IndexType,               max_iteration,      MAX_ITERATION,       NON_NEGATIVE(max_iteration));
	PARAMETER(IndexType,               nupdates,           SPE_NUM_UPDATES,     NON_NEGATIVE(nupdates));
	PARAMETER(bool,                    global_strategy,    SPE_GLOBAL_STRATEGY);
	PARAMETER(bool,                    check_connectivity, CHECK_CONNECTIVITY);

	Neighbors neighbors;
	if (!global_strategy)
	{
		neighbors = find_neighbors(neighbors_method,begin,end,callbacks.distance,k,check_connectivity);
	}

	DO_MEASURE_RUN("SPE");
	STOP_IF_CANCELLED;

	return ReturnResult(spe_embedding(begin,end,callbacks.distance,neighbors,
			target_dimension,global_strategy,tolerance,nupdates,max_iteration), tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(PASS_THRU)
{
	PARAMETER(IndexType, dimension, CURRENT_DIMENSION, POSITIVE(dimension));

	STOP_IF_CANCELLED;

	DenseMatrix feature_matrix(dimension,(NUM_VECTORS));
	DenseVector feature_vector(dimension);
	FeatureVectorCallback feature_vector_callback = callbacks.feature;
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback(*iter,feature_vector);
		feature_matrix.col(iter-begin).array() = feature_vector;
	}
	return ReturnResult(feature_matrix.transpose(),tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(FACTOR_ANALYSIS)
{
	PARAMETER(IndexType,  current_dimension, CURRENT_DIMENSION, POSITIVE(current_dimension));
	PARAMETER(IndexType,  target_dimension,  TARGET_DIMENSION,  IN_RANGE(target_dimension,MINIMAL_TD,NUM_VECTORS));
	PARAMETER(ScalarType, epsilon,           FA_EPSILON,        POSITIVE(epsilon));
	PARAMETER(IndexType,  max_iteration,     MAX_ITERATION,     NON_NEGATIVE(max_iteration));

	DO_MEASURE_RUN("FA");
	STOP_IF_CANCELLED;

	DenseVector mean_vector = compute_mean(begin,end,callbacks.feature,current_dimension);
	return ReturnResult(project(begin,end,callbacks.feature,current_dimension,max_iteration,epsilon,
                                target_dimension, mean_vector), tapkee::ProjectingFunction());
}

IMPLEMENTATION_OF(T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING)
{
	const IndexType N = NUM_VECTORS;
	
	PARAMETER(IndexType,  current_dimension, CURRENT_DIMENSION, POSITIVE(current_dimension));
	PARAMETER(IndexType,  target_dimension,  TARGET_DIMENSION,  EXACTLY(target_dimension,2));
	PARAMETER(ScalarType, perplexity,        SNE_PERPLEXITY,    IN_RANGE(perplexity,0,(N-1)/3));
	PARAMETER(ScalarType, theta,             SNE_THETA,         NON_NEGATIVE(theta));

	DO_MEASURE_RUN("t-SNE");
	STOP_IF_CANCELLED;

	DenseMatrix data(current_dimension,NUM_VECTORS);
	DenseVector feature_vector(current_dimension);
	FeatureVectorCallback feature_vector_callback = callbacks.feature;
	for (RandomAccessIterator iter=begin; iter!=end; ++iter)
	{
		feature_vector_callback(*iter,feature_vector);
		data.col(iter-begin).array() = feature_vector;
	}

	DenseMatrix embedding(target_dimension,N);
	tsne::TSNE* tsne = new tsne::TSNE;
	tsne->run(data.data(),N,current_dimension,embedding.data(),target_dimension,perplexity,theta);
	delete tsne;

	return ReturnResult(embedding.transpose(),tapkee::ProjectingFunction());
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#undef MINIMAL_K
#undef MINIMAL_TD
#undef NUM_VECTORS
#undef DO_MEASURE_RUN
#undef STOP_IF_CANCELLED
#undef IMPLEMENTATION_OF
#undef SKIP_ONE_EIGENVALUE
#undef SKIP_NO_EIGENVALUES
#undef PARAMETER
#undef VA_NUM_ARGS
#undef VA_NUM_ARGS_IMPL_
#undef VA_NUM_ARGS_IMPL
#undef MACRO_DISPATCHER
#undef MACRO_DISPATCHER_
#undef MACRO_DISPATCHER__
#undef MACRO_DISPATCHER___
#undef PARAMETER
#undef PARAMETER3
#undef PARAMETER4
#undef PARAMETER_IMPL
#undef NO_CHECK
#undef IN_RANGE
#undef NOT
#undef POSITIVE
#undef EXACTLY
#undef NON_NEGATIVE

#endif
