#ifndef libedrt_embedding_h_
#define libedrt_embedding_h_

double* diffusion_maps_embedding(
		int N,
		int t,
		int target_dimension,
		double (*kernel)(int, int, const void*),
		const void* user_data);

double* eigendecomposition_embedding(
		double* weight_matrix, 
		int N, 
		int target_dimension,
		bool use_arpack,
		double nullspace_shift);

double* mds_classic_embedding(
		int N,
		int target_dimension,
		double (*distance)(int, int, const void*),
		const void* user_data);

double* lltsa_embedding(
		double* weight_matrix,
		int N,
		int dimension,
		int target_dimension,
		double nullspace_shift,
		void (*obtain_feature_vector)(double*, int, const void*),
		const void* user_data);

#endif
