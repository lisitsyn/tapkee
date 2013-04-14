/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

/* Tapkee includes */
#include <tapkee_defines.hpp>
#include <utils/naming.hpp>
#include <utils/time.hpp>
#include <utils/logging.hpp>
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

namespace {
const ParametersSet defaults = (
	tapkee::keywords::eigen_method = tapkee::keywords::by_default,
	tapkee::keywords::neighbors_method = tapkee::keywords::by_default,
	tapkee::keywords::num_neighbors = tapkee::keywords::by_default,
	tapkee::keywords::target_dimension = tapkee::keywords::by_default,
	tapkee::keywords::diffusion_map_timesteps = tapkee::keywords::by_default,
	tapkee::keywords::gaussian_kernel_width = tapkee::keywords::by_default,
	tapkee::keywords::max_iteration = tapkee::keywords::by_default,
	tapkee::keywords::spe_global_strategy = tapkee::keywords::by_default,
	tapkee::keywords::spe_num_updates = tapkee::keywords::by_default,
	tapkee::keywords::spe_tolerance = tapkee::keywords::by_default,
	tapkee::keywords::landmark_ratio = tapkee::keywords::by_default,
	tapkee::keywords::nullspace_shift = tapkee::keywords::by_default,
	tapkee::keywords::klle_shift = tapkee::keywords::by_default,
	tapkee::keywords::check_connectivity = tapkee::keywords::by_default,
	tapkee::keywords::fa_epsilon = tapkee::keywords::by_default,
	tapkee::keywords::progress_function = tapkee::keywords::by_default,
	tapkee::keywords::cancel_function = tapkee::keywords::by_default,
	tapkee::keywords::sne_perplexity = tapkee::keywords::by_default,
	tapkee::keywords::sne_theta = tapkee::keywords::by_default);
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

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeatureVectorCallback>
class ImplementationBase
{
public:

	ImplementationBase(RandomAccessIterator b, RandomAccessIterator e,
	                   const Callbacks<KernelCallback,DistanceCallback,FeatureVectorCallback>& cbks,
	                   ParametersSet& pmap, const Context& ctx) : 
		parameters(pmap), context(ctx), callbacks(cbks),
		plain_distance(PlainDistance<RandomAccessIterator,DistanceCallback>(cbks.distance)),
		kernel_distance(KernelDistance<RandomAccessIterator,KernelCallback>(cbks.kernel)),
		begin(b), end(e), 
		eigen_method(), neighbors_method(), eigenshift(), traceshift(),
		check_connectivity(), n_neighbors(), width(), timesteps(),
		ratio(), max_iteration(), tolerance(), n_updates(), perplexity(), 
		theta(), global_strategy(), epsilon(), target_dimension(),
		n_vectors(0), current_dimension(0)
	{
		n_vectors = (end-begin);

		target_dimension = parameters(keywords::target_dimension);
		n_neighbors = parameters(keywords::num_neighbors).checked().positive();
		
		if (n_vectors > 0)
		{
			target_dimension = target_dimension.checked().in_range(index(1),index(n_vectors));
			n_neighbors = n_neighbors.checked().in_range(index(3),index(n_vectors));
		}

		eigen_method = parameters(keywords::eigen_method);
		neighbors_method = parameters(keywords::neighbors_method);
		check_connectivity = parameters(keywords::check_connectivity);
		width = parameters(keywords::gaussian_kernel_width).checked().positive();
		timesteps = parameters(keywords::diffusion_map_timesteps).checked().positive();
		eigenshift = parameters(keywords::nullspace_shift);
		traceshift = parameters(keywords::klle_shift);
		max_iteration = parameters(keywords::max_iteration).checked().positive();
		tolerance = parameters(keywords::spe_tolerance).checked().positive();
		n_updates = parameters(keywords::spe_num_updates).checked().positive();
		theta = parameters(keywords::sne_theta).checked().positive();
		global_strategy = parameters(keywords::spe_global_strategy);
		epsilon = parameters(keywords::fa_epsilon);
		perplexity = parameters(keywords::sne_perplexity);
		ratio = parameters(keywords::landmark_ratio);

		if (n_vectors > 0)
		{
			DenseVector dv = DenseVector::Zero(0);
			try 
			{
				callbacks.feature.vector(*begin,dv);
			}
			catch (const unsupported_method_error&)
			{
				current_dimension = 0;
			}
			current_dimension = dv.size();
		}
	}

	static const IndexType SkipOneEigenvalue = 1;
	static const IndexType SkipNoEigenvalues = 0;

	ParametersSet parameters;
	Context context;
	Callbacks<KernelCallback,DistanceCallback,FeatureVectorCallback> callbacks;
	PlainDistance<RandomAccessIterator,DistanceCallback> plain_distance;
	KernelDistance<RandomAccessIterator,KernelCallback> kernel_distance;

	RandomAccessIterator begin;
	RandomAccessIterator end;

	Parameter eigen_method;
	Parameter neighbors_method;
	Parameter eigenshift;
	Parameter traceshift;
	Parameter check_connectivity;
	Parameter n_neighbors;
	Parameter width;
	Parameter timesteps;
	Parameter ratio;
	Parameter max_iteration;
	Parameter tolerance;
	Parameter n_updates;
	Parameter perplexity;
	Parameter theta;
	Parameter global_strategy;
	Parameter epsilon;
	Parameter target_dimension;

	IndexType n_vectors;
	IndexType current_dimension;

	inline static ScalarType scalar(ScalarType v)
	{
		return v;
	}

	inline static IndexType index(IndexType v)
	{
		return v;
	}

	ReturnResult embedUsing(DimensionReductionMethod method)
	{
		if (context.is_cancelled()) 
			throw cancelled_exception();

#define tapkee_method_handle(X)										\
		case X:														\
		{															\
				timed_context tctx__("[+] embedding with " # X);	\
				return embed##X();									\
		}															\
		break														\

		switch (method)
		{
			tapkee_method_handle(KernelLocallyLinearEmbedding);
			tapkee_method_handle(KernelLocalTangentSpaceAlignment);
			tapkee_method_handle(DiffusionMap);
			tapkee_method_handle(MultidimensionalScaling);
			tapkee_method_handle(LandmarkMultidimensionalScaling);
			tapkee_method_handle(Isomap);
			tapkee_method_handle(LandmarkIsomap);
			tapkee_method_handle(NeighborhoodPreservingEmbedding);
			tapkee_method_handle(LinearLocalTangentSpaceAlignment);
			tapkee_method_handle(HessianLocallyLinearEmbedding);
			tapkee_method_handle(LaplacianEigenmaps);
			tapkee_method_handle(LocalityPreservingProjections);
			tapkee_method_handle(PCA);
			tapkee_method_handle(KernelPCA);
			tapkee_method_handle(RandomProjection);
			tapkee_method_handle(StochasticProximityEmbedding);
			tapkee_method_handle(PassThru);
			tapkee_method_handle(FactorAnalysis);
			tapkee_method_handle(tDistributedStochasticNeighborEmbedding);
		}
#undef tapkee_method_handle
		return ReturnResult();
	}

	ReturnResult embedKernelLocallyLinearEmbedding()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_distance,
		                                     n_neighbors,check_connectivity);
		SparseWeightMatrix weight_matrix =
			linear_weight_matrix(begin,end,neighbors,callbacks.kernel,eigenshift,traceshift);
		return ReturnResult(eigen_embedding<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SkipOneEigenvalue).first, tapkee::ProjectingFunction());
	}

	ReturnResult embedKernelLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_distance,
		                                     n_neighbors,check_connectivity);
		SparseWeightMatrix weight_matrix = 
			tangent_weight_matrix(begin,end,neighbors,callbacks.kernel,target_dimension,eigenshift);
		return ReturnResult(eigen_embedding<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SkipOneEigenvalue).first, tapkee::ProjectingFunction());
	}

	ReturnResult embedDiffusionMap()
	{
		#ifdef TAPKEE_GPU
			#define DM_MATRIX_OP GPUDenseImplicitSquareMatrixOperation
		#else
			#define DM_MATRIX_OP DenseImplicitSquareSymmetricMatrixOperation
		#endif

		DenseSymmetricMatrix diffusion_matrix =
			compute_diffusion_matrix(begin,end,callbacks.distance,timesteps,width);
		return ReturnResult(eigen_embedding<DenseSymmetricMatrix,DM_MATRIX_OP>(eigen_method,diffusion_matrix,
			target_dimension,SkipNoEigenvalues).first, tapkee::ProjectingFunction());

		#undef DM_MATRIX_OP
	}

	ReturnResult embedMultidimensionalScaling()
	{
		#ifdef TAPKEE_GPU
			#define MDS_MATRIX_OP GPUDenseImplicitSquareMatrixOperation
		#else
			#define MDS_MATRIX_OP DenseImplicitSquareSymmetricMatrixOperation
		#endif

		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,callbacks.distance);
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EmbeddingResult result = eigen_embedding<DenseSymmetricMatrix,MDS_MATRIX_OP>(eigen_method,
			distance_matrix,target_dimension,SkipNoEigenvalues);

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			result.first.col(i).array() *= sqrt(result.second(i));
		return ReturnResult(result.first, tapkee::ProjectingFunction());
		#undef MDS_MATRIX_OP
	}

	ReturnResult embedLandmarkMultidimensionalScaling()
	{
		ratio = ratio.checked().in_range(scalar(1.0/n_vectors),scalar(1.0 + 1e-6));

		Landmarks landmarks = 
			select_landmarks_random(begin,end,ratio);
		DenseSymmetricMatrix distance_matrix = 
			compute_distance_matrix(begin,end,landmarks,callbacks.distance);
		DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EmbeddingResult landmarks_embedding = 
			eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
					distance_matrix,target_dimension,SkipNoEigenvalues);
		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
		return ReturnResult(triangulate(begin,end,callbacks.distance,landmarks,
			landmark_distances_squared,landmarks_embedding,target_dimension).first, tapkee::ProjectingFunction());
	}

	ReturnResult embedIsomap()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,plain_distance,
		                                     n_neighbors,check_connectivity);
		DenseSymmetricMatrix shortest_distances_matrix = 
			compute_shortest_distances_matrix(begin,end,neighbors,callbacks.distance);
		shortest_distances_matrix = shortest_distances_matrix.array().square();
		centerMatrix(shortest_distances_matrix);
		shortest_distances_matrix.array() *= -0.5;
		
		EmbeddingResult embedding = eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
			shortest_distances_matrix,target_dimension,SkipNoEigenvalues);

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));

		return ReturnResult(embedding.first, tapkee::ProjectingFunction());
	}

	ReturnResult embedLandmarkIsomap()
	{
		ratio = ratio.checked().in_range(scalar(1.0/n_vectors),scalar(1.0 + 1e-6));

		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,plain_distance,
		                                     n_neighbors,check_connectivity);
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
		
		if (eigen_method.is(Dense))
		{
			DenseMatrix distance_matrix_sym = distance_matrix*distance_matrix.transpose();
			landmarks_embedding = eigen_embedding<DenseSymmetricMatrix,DenseImplicitSquareMatrixOperation>
				(eigen_method,distance_matrix_sym,target_dimension,SkipNoEigenvalues);
		}
		else 
		{
			landmarks_embedding = eigen_embedding<DenseSymmetricMatrix,DenseImplicitSquareMatrixOperation>
				(eigen_method,distance_matrix,target_dimension,SkipNoEigenvalues);
		}

		DenseMatrix embedding = distance_matrix.transpose()*landmarks_embedding.first;

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
		return ReturnResult(embedding,tapkee::ProjectingFunction());
	}

	ReturnResult embedNeighborhoodPreservingEmbedding()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_distance,
		                                     n_neighbors,check_connectivity);
		SparseWeightMatrix weight_matrix = 
			linear_weight_matrix(begin,end,neighbors,callbacks.kernel,eigenshift,traceshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
				callbacks.feature,current_dimension);
		EmbeddingResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
				eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SkipNoEigenvalues);
		DenseVector mean_vector = 
			compute_mean(begin,end,callbacks.feature,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,current_dimension),projecting_function);
	}

	ReturnResult embedHessianLocallyLinearEmbedding()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_distance,
		                                     n_neighbors,check_connectivity);
		SparseWeightMatrix weight_matrix =
			hessian_weight_matrix(begin,end,neighbors,callbacks.kernel,target_dimension);
		return ReturnResult(eigen_embedding<SparseWeightMatrix,SparseInverseMatrixOperation>(eigen_method,
			weight_matrix,target_dimension,SkipOneEigenvalue).first, tapkee::ProjectingFunction());
	}

	ReturnResult embedLaplacianEigenmaps()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,plain_distance,
		                                     n_neighbors,check_connectivity);
		Laplacian laplacian = 
			compute_laplacian(begin,end,neighbors,callbacks.distance,width);
		return ReturnResult(generalized_eigen_embedding<SparseWeightMatrix,DenseDiagonalMatrix,SparseInverseMatrixOperation>(
			eigen_method,laplacian.first,laplacian.second,target_dimension,SkipOneEigenvalue).first, tapkee::ProjectingFunction());
	}

	ReturnResult embedLocalityPreservingProjections()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,plain_distance,
		                                     n_neighbors,check_connectivity);
		Laplacian laplacian = 
			compute_laplacian(begin,end,neighbors,callbacks.distance,width);
		DenseSymmetricMatrixPair eigenproblem_matrices =
			construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
					callbacks.feature,current_dimension);
		EmbeddingResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
				eigen_method,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension,SkipNoEigenvalues);
		DenseVector mean_vector = 
			compute_mean(begin,end,callbacks.feature,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,current_dimension),projecting_function);
	}

	ReturnResult embedPCA()
	{
		DenseVector mean_vector = 
			compute_mean(begin,end,callbacks.feature,current_dimension);
		DenseSymmetricMatrix centered_covariance_matrix = 
			compute_covariance_matrix(begin,end,mean_vector,callbacks.feature,current_dimension);
		EmbeddingResult projection_result = 
			eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,centered_covariance_matrix,target_dimension,SkipNoEigenvalues);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,current_dimension), projecting_function);
	}

	ReturnResult embedRandomProjection()
	{
		DenseMatrix projection_matrix = 
			gaussian_projection_matrix(current_dimension, target_dimension);
		DenseVector mean_vector = 
			compute_mean(begin,end,callbacks.feature,current_dimension);

		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_matrix,mean_vector));
		return ReturnResult(project(projection_matrix,mean_vector,begin,end,callbacks.feature,current_dimension), projecting_function);
	}

	ReturnResult embedKernelPCA()
	{
		DenseSymmetricMatrix centered_kernel_matrix = 
			compute_centered_kernel_matrix(begin,end,callbacks.kernel);
		return ReturnResult(eigen_embedding<DenseSymmetricMatrix,DenseMatrixOperation>(eigen_method,
			centered_kernel_matrix,target_dimension,SkipNoEigenvalues).first, tapkee::ProjectingFunction());
	}

	ReturnResult embedLinearLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = find_neighbors(neighbors_method,begin,end,kernel_distance,
		                                     n_neighbors,check_connectivity);
		SparseWeightMatrix weight_matrix = 
			tangent_weight_matrix(begin,end,neighbors,callbacks.kernel,target_dimension,eigenshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_lltsa_eigenproblem(weight_matrix,begin,end,
				callbacks.feature,current_dimension);
		EmbeddingResult projection_result = 
			generalized_eigen_embedding<DenseSymmetricMatrix,DenseSymmetricMatrix,DenseInverseMatrixOperation>(
				eigen_method,eig_matrices.first,eig_matrices.second,target_dimension,SkipNoEigenvalues);
		DenseVector mean_vector = 
			compute_mean(begin,end,callbacks.feature,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return ReturnResult(project(projection_result.first,mean_vector,begin,end,callbacks.feature,current_dimension),
				projecting_function);
	}

	ReturnResult embedStochasticProximityEmbedding()
	{
		Neighbors neighbors;
		if (global_strategy.is(false))
		{
			Neighbors neighbors = find_neighbors(neighbors_method,begin,end,plain_distance,
			                                     n_neighbors,check_connectivity);
		}

		return ReturnResult(spe_embedding(begin,end,callbacks.distance,neighbors,
				target_dimension,global_strategy,tolerance,n_updates,max_iteration), tapkee::ProjectingFunction());
	}

	ReturnResult embedPassThru()
	{
		DenseMatrix feature_matrix(static_cast<IndexType>(current_dimension),n_vectors);
		DenseVector feature_vector(static_cast<IndexType>(current_dimension));
		FeatureVectorCallback feature_vector_callback = callbacks.feature;
		for (RandomAccessIterator iter=begin; iter!=end; ++iter)
		{
			feature_vector_callback.vector(*iter,feature_vector);
			feature_matrix.col(iter-begin).array() = feature_vector;
		}
		return ReturnResult(feature_matrix.transpose(),tapkee::ProjectingFunction());
	}

	ReturnResult embedFactorAnalysis()
	{
		DenseVector mean_vector = compute_mean(begin,end,callbacks.feature,current_dimension);
		return ReturnResult(project(begin,end,callbacks.feature,current_dimension,max_iteration,epsilon,
									target_dimension, mean_vector), tapkee::ProjectingFunction());
	}

	ReturnResult embedtDistributedStochasticNeighborEmbedding()
	{
		perplexity = perplexity.checked().in_range(scalar(0.0),scalar((n_vectors-1)/3.0 + 1e-6));

		DenseMatrix data(static_cast<IndexType>(current_dimension),n_vectors);
		DenseVector feature_vector(static_cast<IndexType>(current_dimension));
		FeatureVectorCallback feature_vector_callback = callbacks.feature;
		for (RandomAccessIterator iter=begin; iter!=end; ++iter)
		{
			feature_vector_callback.vector(*iter,feature_vector);
			data.col(iter-begin).array() = feature_vector;
		}

		DenseMatrix embedding(static_cast<IndexType>(target_dimension),n_vectors);
		tsne::TSNE* tsne = new tsne::TSNE;
		tsne->run(data.data(),n_vectors,current_dimension,embedding.data(),target_dimension,perplexity,theta);
		delete tsne;

		return ReturnResult(embedding.transpose(),tapkee::ProjectingFunction());
	}

};

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeatureVectorCallback>
ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback>
	initialize(RandomAccessIterator begin, RandomAccessIterator end,
	           KernelCallback kernel, DistanceCallback distance, FeatureVectorCallback feature_vector,
	           ParametersSet& pmap, const Context& ctx)
{
	return ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeatureVectorCallback>(
			begin,end,Callbacks<KernelCallback,DistanceCallback,FeatureVectorCallback>(kernel,distance,feature_vector),
			pmap,ctx);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
