/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHODS_H_
#define TAPKEE_METHODS_H_

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/utils/naming.hpp>
#include <tapkee/utils/time.hpp>
#include <tapkee/utils/logging.hpp>
#include <tapkee/utils/conditional_select.hpp>
#include <tapkee/utils/features.hpp>
#include <tapkee/parameters/defaults.hpp>
#include <tapkee/parameters/context.hpp>
#include <tapkee/routines/locally_linear.hpp>
#include <tapkee/routines/eigendecomposition.hpp>
#include <tapkee/routines/generalized_eigendecomposition.hpp>
#include <tapkee/routines/multidimensional_scaling.hpp>
#include <tapkee/routines/diffusion_maps.hpp>
#include <tapkee/routines/laplacian_eigenmaps.hpp>
#include <tapkee/routines/isomap.hpp>
#include <tapkee/routines/pca.hpp>
#include <tapkee/routines/random_projection.hpp>
#include <tapkee/routines/spe.hpp>
#include <tapkee/routines/fa.hpp>
#include <tapkee/routines/manifold_sculpting.hpp>
#include <tapkee/neighbors/neighbors.hpp>
#include <tapkee/external/barnes_hut_sne/tsne.hpp>
/* End of Tapkee includes */

namespace tapkee
{
//! Main namespace for all internal routines, should not be exposed as public API
namespace tapkee_internal
{

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeaturesCallback>
class ImplementationBase
{
public:

	ImplementationBase(RandomAccessIterator b, RandomAccessIterator e,
	                   KernelCallback k, DistanceCallback d, FeaturesCallback f,
	                   stichwort::ParametersSet& pmap, const Context& ctx) : 
		parameters(pmap), context(ctx), kernel(k), distance(d), features(f),
		plain_distance(PlainDistance<RandomAccessIterator,DistanceCallback>(distance)),
		kernel_distance(KernelDistance<RandomAccessIterator,KernelCallback>(kernel)),
		begin(b), end(e), computation_strategy(),
		eigen_method(), neighbors_method(), eigenshift(), traceshift(),
		check_connectivity(), n_neighbors(), width(), timesteps(),
		ratio(), max_iteration(), tolerance(), n_updates(), perplexity(), 
		theta(), squishing_rate(), global_strategy(), epsilon(), target_dimension(),
		n_vectors(0), current_dimension(0)
	{
		n_vectors = (end-begin);

		target_dimension = parameters(target_dimension);
		n_neighbors = parameters(num_neighbors).checked().positive();
		
		if (n_vectors > 0)
		{
			target_dimension.checked()
				.inRange(static_cast<IndexType>(1),static_cast<IndexType>(n_vectors));
			n_neighbors.checked()
				.inRange(static_cast<IndexType>(3),static_cast<IndexType>(n_vectors));
		}

		computation_strategy = parameters(computation_strategy);
		eigen_method = parameters(eigen_method);
		neighbors_method = parameters(neighbors_method);
		check_connectivity = parameters(check_connectivity);
		width = parameters(gaussian_kernel_width).checked().positive();
		timesteps = parameters(diffusion_map_timesteps).checked().positive();
		eigenshift = parameters(nullspace_shift);
		traceshift = parameters(klle_shift);
		max_iteration = parameters(max_iteration);
		tolerance = parameters(spe_tolerance).checked().positive();
		n_updates = parameters(spe_num_updates).checked().positive();
		theta = parameters(sne_theta).checked().nonNegative();
		squishing_rate = parameters(squishing_rate);
		global_strategy = parameters(spe_global_strategy);
		epsilon = parameters(fa_epsilon).checked().nonNegative();
		perplexity = parameters(sne_perplexity).checked().nonNegative();
		ratio = parameters(landmark_ratio);

		if (!is_dummy<FeaturesCallback>::value)
		{
			current_dimension = features.dimension();
		}
		else
		{
			current_dimension = 0;
		}
	}

	TapkeeOutput embedUsing(DimensionReductionMethod method)
	{
		if (context.is_cancelled()) 
			throw cancelled_exception();

		using std::mem_fun_ref_t;
		using std::mem_fun_ref;
		typedef std::mem_fun_ref_t<TapkeeOutput,ImplementationBase> ImplRef;

#define tapkee_method_handle(X)																	\
		case X:																					\
		{																						\
			timed_context tctx__("[+] embedding with " # X);									\
			ImplRef ref = conditional_select<													\
				((!MethodTraits<X>::needs_kernel)   || (!is_dummy<KernelCallback>::value))   &&	\
				((!MethodTraits<X>::needs_distance) || (!is_dummy<DistanceCallback>::value)) &&	\
				((!MethodTraits<X>::needs_features) || (!is_dummy<FeaturesCallback>::value)),	\
					ImplRef>()(mem_fun_ref(&ImplementationBase::embed##X),						\
					           mem_fun_ref(&ImplementationBase::embedEmpty));					\
			return ref(*this);																	\
		}																						\
		break																					\

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
			tapkee_method_handle(ManifoldSculpting);
		}
#undef tapkee_method_handle
		return TapkeeOutput();
	}

private:

	stichwort::ParametersSet parameters;
	Context context;
	KernelCallback kernel;
	DistanceCallback distance;
	FeaturesCallback features;
	PlainDistance<RandomAccessIterator,DistanceCallback> plain_distance;
	KernelDistance<RandomAccessIterator,KernelCallback> kernel_distance;

	RandomAccessIterator begin;
	RandomAccessIterator end;

	stichwort::Parameter computation_strategy;
	stichwort::Parameter eigen_method;
	stichwort::Parameter neighbors_method;
	stichwort::Parameter eigenshift;
	stichwort::Parameter traceshift;
	stichwort::Parameter check_connectivity;
	stichwort::Parameter n_neighbors;
	stichwort::Parameter width;
	stichwort::Parameter timesteps;
	stichwort::Parameter ratio;
	stichwort::Parameter max_iteration;
	stichwort::Parameter tolerance;
	stichwort::Parameter n_updates;
	stichwort::Parameter perplexity;
	stichwort::Parameter theta;
	stichwort::Parameter squishing_rate;	
	stichwort::Parameter global_strategy;
	stichwort::Parameter epsilon;
	stichwort::Parameter target_dimension;

	IndexType n_vectors;
	IndexType current_dimension;

	template<class Distance>
	Neighbors findNeighborsWith(Distance d)
	{
		return find_neighbors(neighbors_method,begin,end,d,n_neighbors,check_connectivity);
	}

	static tapkee::ProjectingFunction unimplementedProjectingFunction() 
	{
		return tapkee::ProjectingFunction();
	}

	TapkeeOutput embedEmpty()
	{
		throw unsupported_method_error("Some callback is missed");
		return TapkeeOutput();
	}

	TapkeeOutput embedKernelLocallyLinearEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			linear_weight_matrix(begin,end,neighbors,kernel,eigenshift,traceshift);
		DenseMatrix embedding = 
			eigendecomposition(eigen_method,computation_strategy,SmallestEigenvalues,
					weight_matrix,target_dimension).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedKernelLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix = 
			tangent_weight_matrix(begin,end,neighbors,kernel,target_dimension,eigenshift);
		DenseMatrix embedding =
			eigendecomposition(eigen_method,computation_strategy,SmallestEigenvalues,
					weight_matrix,target_dimension).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedDiffusionMap()
	{
		DenseSymmetricMatrix diffusion_matrix =
			compute_diffusion_matrix(begin,end,distance,timesteps,width);
		DenseMatrix embedding =
			eigendecomposition(eigen_method,computation_strategy,SquaredLargestEigenvalues,
					diffusion_matrix,target_dimension).first;

		return TapkeeOutput(embedding, unimplementedProjectingFunction());
	}

	TapkeeOutput embedMultidimensionalScaling()
	{
		DenseSymmetricMatrix distance_matrix = compute_distance_matrix(begin,end,distance);
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EigendecompositionResult embedding = 
			eigendecomposition(eigen_method,computation_strategy,LargestEigenvalues,
					distance_matrix,target_dimension);

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
		#undef MDS_MATRIX_OP
	}

	TapkeeOutput embedLandmarkMultidimensionalScaling()
	{
		ratio.checked()
			.inClosedRange(static_cast<ScalarType>(3.0/n_vectors),
			               static_cast<ScalarType>(1.0));

		Landmarks landmarks = 
			select_landmarks_random(begin,end,ratio);
		DenseSymmetricMatrix distance_matrix = 
			compute_distance_matrix(begin,end,landmarks,distance);
		DenseVector landmark_distances_squared = distance_matrix.colwise().mean();
		centerMatrix(distance_matrix);
		distance_matrix.array() *= -0.5;
		EigendecompositionResult landmarks_embedding = 
			eigendecomposition(eigen_method,computation_strategy,LargestEigenvalues,
					distance_matrix,target_dimension);
		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			landmarks_embedding.first.col(i).array() *= sqrt(landmarks_embedding.second(i));
		return TapkeeOutput(triangulate(begin,end,distance,landmarks,
			landmark_distances_squared,landmarks_embedding,target_dimension), unimplementedProjectingFunction());
	}

	TapkeeOutput embedIsomap()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		DenseSymmetricMatrix shortest_distances_matrix = 
			compute_shortest_distances_matrix(begin,end,neighbors,distance);
		shortest_distances_matrix = shortest_distances_matrix.array().square();
		centerMatrix(shortest_distances_matrix);
		shortest_distances_matrix.array() *= -0.5;
		
		EigendecompositionResult embedding = 
			eigendecomposition(eigen_method,computation_strategy,LargestEigenvalues,
					shortest_distances_matrix,target_dimension);

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));

		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLandmarkIsomap()
	{
		ratio.checked()
			.inClosedRange(static_cast<ScalarType>(3.0/n_vectors),
			               static_cast<ScalarType>(1.0));

		Neighbors neighbors = findNeighborsWith(plain_distance);
		Landmarks landmarks = 
			select_landmarks_random(begin,end,ratio);
		DenseMatrix distance_matrix = 
			compute_shortest_distances_matrix(begin,end,landmarks,neighbors,distance);
		distance_matrix = distance_matrix.array().square();
		
		DenseVector col_means = distance_matrix.colwise().mean();
		DenseVector row_means = distance_matrix.rowwise().mean();
		ScalarType grand_mean = distance_matrix.mean();
		distance_matrix.array() += grand_mean;
		distance_matrix.colwise() -= row_means;
		distance_matrix.rowwise() -= col_means.transpose();
		distance_matrix.array() *= -0.5;

		EigendecompositionResult landmarks_embedding;
		
		if (eigen_method.is(Dense))
		{
			DenseMatrix distance_matrix_sym = distance_matrix*distance_matrix.transpose();
			landmarks_embedding = eigendecomposition(eigen_method,computation_strategy,
					LargestEigenvalues,distance_matrix_sym,target_dimension);
		}
		else 
		{
			landmarks_embedding = eigendecomposition(eigen_method,computation_strategy,
					SquaredLargestEigenvalues,distance_matrix,target_dimension);
		}

		DenseMatrix embedding = distance_matrix.transpose()*landmarks_embedding.first;

		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.col(i).array() /= sqrt(sqrt(landmarks_embedding.second(i)));
		return TapkeeOutput(embedding,unimplementedProjectingFunction());
	}

	TapkeeOutput embedNeighborhoodPreservingEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix = 
			linear_weight_matrix(begin,end,neighbors,kernel,eigenshift,traceshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_neighborhood_preserving_eigenproblem(weight_matrix,begin,end,
				features,current_dimension);
		EigendecompositionResult projection_result = 
			generalized_eigendecomposition(eigen_method,computation_strategy,
					SmallestEigenvalues,eig_matrices.first,eig_matrices.second,target_dimension);
		DenseVector mean_vector = 
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension),projecting_function);
	}

	TapkeeOutput embedHessianLocallyLinearEmbedding()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix =
			hessian_weight_matrix(begin,end,neighbors,kernel,target_dimension);
		return TapkeeOutput(eigendecomposition(eigen_method,computation_strategy,
					SmallestEigenvalues,weight_matrix,target_dimension).first, 
				unimplementedProjectingFunction());
	}

	TapkeeOutput embedLaplacianEigenmaps()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		Laplacian laplacian = 
			compute_laplacian(begin,end,neighbors,distance,width);
		return TapkeeOutput(generalized_eigendecomposition(eigen_method,computation_strategy,
					SmallestEigenvalues,laplacian.first,laplacian.second,target_dimension).first,
				unimplementedProjectingFunction());
	}

	TapkeeOutput embedLocalityPreservingProjections()
	{
		Neighbors neighbors = findNeighborsWith(plain_distance);
		Laplacian laplacian = 
			compute_laplacian(begin,end,neighbors,distance,width);
		DenseSymmetricMatrixPair eigenproblem_matrices =
			construct_locality_preserving_eigenproblem(laplacian.first,laplacian.second,begin,end,
					features,current_dimension);
		EigendecompositionResult projection_result = 
			generalized_eigendecomposition(eigen_method,computation_strategy,
					SmallestEigenvalues,eigenproblem_matrices.first,eigenproblem_matrices.second,target_dimension);
		DenseVector mean_vector = 
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedPCA()
	{
		DenseVector mean_vector = 
			compute_mean(begin,end,features,current_dimension);
		DenseSymmetricMatrix centered_covariance_matrix = 
			compute_covariance_matrix(begin,end,mean_vector,features,current_dimension);
		EigendecompositionResult projection_result = 
			eigendecomposition(eigen_method,computation_strategy,
					LargestEigenvalues,centered_covariance_matrix,target_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedRandomProjection()
	{
		DenseMatrix projection_matrix = 
			gaussian_projection_matrix(current_dimension, target_dimension);
		DenseVector mean_vector = 
			compute_mean(begin,end,features,current_dimension);

		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_matrix,mean_vector));
		return TapkeeOutput(project(projection_matrix,mean_vector,begin,end,features,current_dimension), projecting_function);
	}

	TapkeeOutput embedKernelPCA()
	{
		DenseSymmetricMatrix centered_kernel_matrix = 
			compute_centered_kernel_matrix(begin,end,kernel);
		EigendecompositionResult embedding = eigendecomposition(eigen_method,computation_strategy,
				LargestEigenvalues,centered_kernel_matrix,target_dimension);
		for (IndexType i=0; i<static_cast<IndexType>(target_dimension); i++)
			embedding.first.col(i).array() *= sqrt(embedding.second(i));
		return TapkeeOutput(embedding.first, unimplementedProjectingFunction());
	}

	TapkeeOutput embedLinearLocalTangentSpaceAlignment()
	{
		Neighbors neighbors = findNeighborsWith(kernel_distance);
		SparseWeightMatrix weight_matrix = 
			tangent_weight_matrix(begin,end,neighbors,kernel,target_dimension,eigenshift);
		DenseSymmetricMatrixPair eig_matrices =
			construct_lltsa_eigenproblem(weight_matrix,begin,end,
				features,current_dimension);
		EigendecompositionResult projection_result = 
			generalized_eigendecomposition(eigen_method,computation_strategy,SmallestEigenvalues,
					eig_matrices.first,eig_matrices.second,target_dimension);
		DenseVector mean_vector = 
			compute_mean(begin,end,features,current_dimension);
		tapkee::ProjectingFunction projecting_function(new tapkee::MatrixProjectionImplementation(projection_result.first,mean_vector));
		return TapkeeOutput(project(projection_result.first,mean_vector,begin,end,features,current_dimension),
				projecting_function);
	}

	TapkeeOutput embedStochasticProximityEmbedding()
	{
		Neighbors neighbors;
		if (global_strategy.is(false))
		{
			neighbors = findNeighborsWith(plain_distance);
		}

		return TapkeeOutput(spe_embedding(begin,end,distance,neighbors,
				target_dimension,global_strategy,tolerance,n_updates,max_iteration), unimplementedProjectingFunction());
	}

	TapkeeOutput embedPassThru()
	{
		DenseMatrix feature_matrix =
			dense_matrix_from_features(features, current_dimension, begin, end);
		return TapkeeOutput(feature_matrix.transpose(),tapkee::ProjectingFunction());
	}

	TapkeeOutput embedFactorAnalysis()
	{
		DenseVector mean_vector = compute_mean(begin,end,features,current_dimension);
		return TapkeeOutput(project(begin,end,features,current_dimension,max_iteration,epsilon,
									target_dimension, mean_vector), unimplementedProjectingFunction());
	}

	TapkeeOutput embedtDistributedStochasticNeighborEmbedding()
	{
		perplexity.checked()
			.inClosedRange(static_cast<ScalarType>(0.0),
			               static_cast<ScalarType>((n_vectors-1)/3.0));

		DenseMatrix data = 
			dense_matrix_from_features(features, current_dimension, begin, end);

		DenseMatrix embedding(static_cast<IndexType>(target_dimension),n_vectors);
		tsne::TSNE tsne;
		tsne.run(data.data(),n_vectors,current_dimension,embedding.data(),target_dimension,perplexity,theta);

		return TapkeeOutput(embedding.transpose(), unimplementedProjectingFunction());
	}

	TapkeeOutput embedManifoldSculpting()
	{
		squishing_rate.checked()
			.inRange(static_cast<ScalarType>(0.0),
			         static_cast<ScalarType>(1.0));

		DenseMatrix embedding =
			dense_matrix_from_features(features, current_dimension, begin, end);

		Neighbors neighbors = findNeighborsWith(plain_distance);

		manifold_sculpting_embed(begin, end, embedding, target_dimension, neighbors, distance, max_iteration, squishing_rate);

		return TapkeeOutput(embedding, tapkee::ProjectingFunction());
	}

};

template <class RandomAccessIterator, class KernelCallback,
          class DistanceCallback, class FeaturesCallback>
ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeaturesCallback>
	initialize(RandomAccessIterator begin, RandomAccessIterator end,
	           KernelCallback kernel, DistanceCallback distance, FeaturesCallback features,
	           stichwort::ParametersSet& pmap, const Context& ctx)
{
	return ImplementationBase<RandomAccessIterator,KernelCallback,DistanceCallback,FeaturesCallback>(
			begin,end,kernel,distance,features,pmap,ctx);
}

} // End of namespace tapkee_internal
} // End of namespace tapkee

#endif
