#include "libedrt_methods.hpp"
#include <vector>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string.h>

double* diffusion_maps_embedding(
		int N,
		int t,
		int target_dimension,
		double (*kernel)(int, int, const void*),
		const void* user_data)
{
#ifdef HAVE_ARPACK
	bool use_arpack = true;
#else
	bool use_arpack = false;
#endif
	int32_t i,j;

	double* kernel_matrix;// = SG_MALLOC(double, N*N);
	for (i=0; i<N; i++)
	{
		for (j=i; j<N; j++)
		{
			double kernel_value = kernel(i,j,user_data);
			kernel_matrix[i*N+j] = kernel_value;
			kernel_matrix[j*N+i] = kernel_value;
		}
	}

	double* p_vector;// = SG_CALLOC(double, N);
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			p_vector[i] += kernel_matrix[j*N+i];
		}
	}
	//CMath::display_matrix(kernel_matrix.matrix,N,N,"K");
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			kernel_matrix[i*N+j] /= pow(p_vector[i]*p_vector[j], t);
		}
	}
	//CMath::display_matrix(kernel_matrix.matrix,N,N,"K");

	for (i=0; i<N; i++)
	{
		p_vector[i] = 0.0;
		for (j=0; j<N; j++)
		{
			p_vector[i] += kernel_matrix[j*N+i];
		}
		p_vector[i] = sqrt(p_vector[i]);
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			kernel_matrix[i*N+j] /= p_vector[i]*p_vector[j];
		}
	}

	double* s_values = p_vector;

	int32_t info = 0;
	
	double* new_feature_matrix;// = SG_MALLOC(double, N*target_dimension);

	if (use_arpack)
	{
#ifdef HAVE_ARPACK
		arpack_dsxupd(kernel_matrix,NULL,false,N,target_dimension,"LA",false,1,false,true,0.0,0.0,s_values,kernel_matrix,info);
#endif /* HAVE_ARPACK */
		for (i=0; i<target_dimension; i++)
		{
			for (j=0; j<N; j++)
				new_feature_matrix[j*target_dimension+i] = kernel_matrix[j*target_dimension+i];
		}
	}
	else 
	{
		//SG_SWARNING("LAPACK does not provide efficient routines to construct embedding (this may take time). Consider installing ARPACK.");
		//wrap_dgesvd('O','N',N,N,kernel_matrix,N,s_values,NULL,1,NULL,1,&info);
		for (i=0; i<target_dimension; i++)
		{
			for (j=0; j<N; j++)
				new_feature_matrix[j*target_dimension+i] = 
				    kernel_matrix[(target_dimension-i-1)*N+j];
		}
	}
	//if (info)
	//	SG_SERROR("Eigenproblem solving failed with %d code", info);

	//SG_FREE(kernel_matrix);
	//SG_FREE(s_values);

	return new_feature_matrix;
}

double* eigendecomposition_embedding(
		double* weight_matrix, 
		int N, 
		int target_dimension,
		bool use_arpack,
		double nullspace_shift)
{
	int i,j;
	// get eigenvectors with ARPACK or LAPACK
	int eigenproblem_status = 0;

	double* eigenvalues_vector = NULL;
	double* eigenvectors = NULL;
	double* embedding_feature_matrix = NULL;
	if (use_arpack)
	{
#ifndef HAVE_ARPACK
		//SG_ERROR("ARPACK is not supported in this configuration.\n");
#endif
		// using ARPACK (faster)
		//eigenvalues_vector;// = SG_MALLOC(double, target_dimension+1);
#ifdef HAVE_ARPACK
		shogun::arpack_dsxupd(weight_matrix, NULL, false, N, target_dimension+1,
		                      "LA", true, 3, true, false, nullspace_shift, 0.0,
		                      eigenvalues_vector, weight_matrix, eigenproblem_status);
#endif
		//embedding_feature_matrix = SG_MALLOC(double, N*target_dimension);
		for (i=0; i<target_dimension; i++)
		{
			for (j=0; j<N; j++)
				embedding_feature_matrix[j*target_dimension+i] = 
				    weight_matrix[j*(target_dimension+1)+i+1];
		}
		//SG_FREE(eigenvalues_vector);
	}
	else
	{
		// using LAPACK (slower)
		//eigenvalues_vector = SG_MALLOC(double, N);
		//eigenvectors = SG_MALLOC(double, (target_dimension+1)*N);
		//shogun::wrap_dsyevr('V','U',N,weight_matrix,N,2,target_dimension+2,
		//                    eigenvalues_vector,eigenvectors,&eigenproblem_status);
		//embedding_feature_matrix = SG_MALLOC(double, N*target_dimension);
		// LAPACKed eigenvectors
		for (i=0; i<target_dimension; i++)
		{
			for (j=0; j<N; j++)
				embedding_feature_matrix[j*target_dimension+i] = eigenvectors[i*N+j];
		}
		//SG_FREE(eigenvectors);
		//SG_FREE(eigenvalues_vector);
	}
	return embedding_feature_matrix;
}

double* mds_classic_embedding(int N,
                              int target_dimension, 
                              double (*distance)(int, int, const void*),
                              const void* user_data)
{
	// loop variables
	int32_t i,j;

	// double center distance_matrix
	double dsq;

	double* distance_matrix;// = SG_MALLOC(double, N*N);
	for (i=0; i<N; i++)
	{
		for (j=i; j<N; j++)
		{
			//dsq = CMath::sq(distance(i,j,user_data));
			distance_matrix[i*N+j] = dsq;
			distance_matrix[j*N+i] = dsq;
		}
	}
	//CMath::center_matrix(distance_matrix,N,N);
	for (i=0; i<N; i++)
	{
		distance_matrix[i*N+i] *= -0.5;
		for (j=i+1; j<N; j++)
		{
			distance_matrix[i*N+j] *= -0.5;
			distance_matrix[j*N+i] *= -0.5;
		}
	}

	// feature matrix representing given distance
	double* embedding_feature_matrix;// = SG_MALLOC(double, N*target_dimension);

	// status of eigenproblem to be solved
	int eigenproblem_status = 0;
#ifdef HAVE_ARPACK
	// using ARPACK
	double* eigenvalues_vector;// = SG_MALLOC(double, target_dimension);
	// solve eigenproblem with ARPACK (faster)
	arpack_dsxupd(distance_matrix,NULL,false,N,target_dimension,"LM",false,1,false,false,0.0,0.0,
	              eigenvalues_vector,embedding_feature_matrix,eigenproblem_status);
	// check for failure
	ASSERT(eigenproblem_status == 0);
	// reverse eigenvectors order
	double tmp;
	for (j=0; j<N; j++)
	{
		for (i=0; i<target_dimension/2; i++)
		{
			tmp = embedding_feature_matrix[j*target_dimension+i];
			embedding_feature_matrix[j*target_dimension+i] =
				embedding_feature_matrix[j*target_dimension+(target_dimension-i-1)];
			embedding_feature_matrix[j*target_dimension+(target_dimension-i-1)] = tmp;
		}
	}
	// reverse eigenvalues order
	for (i=0; i<target_dimension/2; i++)
	{
		tmp = eigenvalues_vector[i];
		eigenvalues_vector[i] = eigenvalues_vector[target_dimension-i-1];
		eigenvalues_vector[target_dimension-i-1] = tmp;
	}

	// finally construct embedding
	for (i=0; i<target_dimension; i++)
	{
		for (j=0; j<N; j++)
			embedding_feature_matrix[j*target_dimension+i] *=
				CMath::sqrt(eigenvalues_vector[i]);
	}

#else /* not HAVE_ARPACK */
	// using LAPACK
	double* eigenvalues_vector;// = SG_MALLOC(double, N);
	double* eigenvectors;// = SG_MALLOC(double, target_dimension*N);
	// solve eigenproblem with LAPACK
	//wrap_dsyevr('V','U',N,distance_matrix,N,N-target_dimension+1,
	  //          N,eigenvalues_vector,eigenvectors,&eigenproblem_status);
	// check for failure
	//ASSERT(eigenproblem_status==0);

	// construct embedding
	for (i=0; i<target_dimension; i++)
	{
		for (j=0; j<N; j++)
		{
			embedding_feature_matrix[j*target_dimension+i] =
			      eigenvectors[(target_dimension-i-1)*N+j] * sqrt(eigenvalues_vector[i]);
		}
	}
	//SG_FREE(eigenvalues_vector);
	//SG_FREE(eigenvectors);
#endif /* HAVE_ARPACK else */

	return embedding_feature_matrix;
}

double* lltsa_embedding(
		double* weight_matrix,
		int N,
		int dimension,
		int target_dimension,
		double nullspace_shift,
		void (*obtain_feature_vector)(double*, int, const void*),
		const void* user_data
		)
{
	int i,j;

	double* feature_matrix;// = SG_MALLOC(double, dimension*N);
	for (i=0; i<N; i++)
		obtain_feature_vector(feature_matrix+i*dimension, i, user_data);

	double* XTM;// = SG_MALLOC(double, dimension*N);
	double* lhs_M;// = SG_MALLOC(double, dimension*dimension);
	double* rhs_M;// = SG_MALLOC(double, dimension*dimension);
	//CMath::center_matrix(weight_matrix,N,N);

	/*
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
	            dimension,N,N,
	            1.0,feature_matrix,dimension,
	                weight_matrix,N,
	            0.0,XTM,dimension);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
	            dimension,dimension,N,
	            1.0,XTM,dimension,
	                feature_matrix,dimension,
	            0.0,lhs_M,dimension);
	*/
//	double* mean_vector = SG_CALLOC(double, dimension);
//	for (i=0; i<N; i++)
//		cblas_daxpy(dimension,1.0,feature_matrix+i*dimension,1,mean_vector,1);

//	cblas_dscal(dimension,1.0/N,mean_vector,1);

//	for (i=0; i<N; i++)
//		cblas_daxpy(dimension,-1.0,mean_vector,1,feature_matrix+i*dimension,1);

//	SG_FREE(mean_vector);

//	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
//	            dimension,dimension,N,
//	            1.0,feature_matrix,dimension,
//	                feature_matrix,dimension,
//	            0.0,rhs_M,dimension);

	for (i=0; i<dimension; i++) rhs_M[i*dimension+i] += 1e-6;

	double* evals;// = SG_MALLOC(double, dimension);
	double* evectors;// = SG_MALLOC(double, dimension*dimension);
	int32_t info = 0;
#ifdef HAVE_ARPACK
	arpack_dsxupd(lhs_M,rhs_M,false,dimension,dimension,"LA",false,3,true,false,nullspace_shift,0.0,
	              evals,evectors,info);
#else
//	wrap_dsygvx(1,'V','U',dimension,lhs_M,dimension,rhs_M,dimension,
//	            dimension-target_dimension+1,dimension,evals,evectors,&info);
#endif

//	SG_FREE(lhs_M);
//	SG_FREE(rhs_M);
//	SG_FREE(evals);
//	if (info!=0) 
//		SG_SERROR("Failed to solve eigenproblem (%d)\n",info);

//	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
//	            N,target_dimension,dimension,
//	            1.0,feature_matrix,dimension,
//	                evectors,target_dimension,
//	            0.0,XTM,N);
//	SG_FREE(evectors);
//	SG_FREE(feature_matrix);

	double* embedding_feature_matrix;// = SG_MALLOC(double, target_dimension*N);
	for (i=0; i<target_dimension; i++)
	{
		for (j=0; j<N; j++)
			embedding_feature_matrix[j*dimension+i] = XTM[i*N+j];
	}
	//SG_FREE(XTM);
	return embedding_feature_matrix;
}
