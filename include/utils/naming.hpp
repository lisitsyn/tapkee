/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_NAMING_H_
#define TAPKEE_NAMING_H_

namespace tapkee
{

string get_method_name(MethodId m)
{
	switch (m)
	{
		case KernelLocallyLinearEmbedding: return "Kernel Locally Linear Embedding";
		case KernelLocalTangentSpaceAlignment: return "Local Tangent Space Alignment";
		case DiffusionMap: return "Diffusion Map";
		case MultidimensionalScaling: return "Classic Multidimensional Scaling";
		case LandmarkMultidimensionalScaling: return "Landmark Multidimensional Scaling";
		case Isomap: return "Isomap";
		case LandmarkIsomap: return "Landmark Isomap";
		case NeighborhoodPreservingEmbedding: return "Neighborhood Preserving Embedding";
		case LinearLocalTangentSpaceAlignment: return "Linear Local Tangent Space Alignment";
		case HessianLocallyLinearEmbedding: return "Hessian Locally Linear Embedding";
		case LaplacianEigenmaps: return "Laplacian Eigenmaps";
		case LocalityPreservingProjections: return "Locality Preserving Embedding";
		case PCA: return "Principal Component Analysis";
		case KernelPCA: return "Kernel Principal Component Analysis";
		case StochasticProximityEmbedding: return "Stochastic Proximity Embedding";
		case PassThru: return "passing through";
		case RandomProjection: return "Random Projection";
		case FactorAnalysis: return "Factor Analysis";
		case tDistributedStochasticNeighborEmbedding: return "t-distributed Stochastic Neighbor Embedding";
	}
	return "hello";
}

string get_parameter_name(ParameterKey pk)
{
	switch (pk)
	{
#define __parameter_case(X) case X: return ""#X; break
		__parameter_case(ReductionMethod);
		__parameter_case(NumberOfNeighbors);
		__parameter_case(TargetDimension);
		__parameter_case(CurrentDimension);
		__parameter_case(EigenEmbeddingMethod);
		__parameter_case(NeighborsMethod);
		__parameter_case(DiffusionMapTimesteps);
		__parameter_case(GaussianKernelWidth);
		__parameter_case(MaxIteration);
		__parameter_case(SpeGlobalStrategy);
		__parameter_case(SpeNumberOfUpdates);
		__parameter_case(SpeTolerance);
		__parameter_case(LandmarkRatio);
		__parameter_case(NullspaceShift);
		__parameter_case(KlleShift);
		__parameter_case(CheckConnectivity);
		__parameter_case(FaEpsilon);
		__parameter_case(ProgressFunction);
		__parameter_case(CancelFunction);
		__parameter_case(SnePerplexity);
		__parameter_case(SneTheta);
#undef parameter_case
	}
	return "hello";
}

string get_neighbors_method_name(NeighborsMethodId m)
{
	switch (m)
	{
		case Brute: return "Brute-force";
		case CoverTree: return "Cover Tree";
	}
	return "hello";
}

string get_eigenembedding_method_name(EigenEmbeddingMethodId m)
{
	switch (m)
	{
#ifdef TAPKEE_WITH_ARPACK
		case Arpack: return "Arpack";
#endif
		case Dense: return "Dense";
		case Randomized: return "Randomized";
	}
	return "hello";
}

}
#endif
