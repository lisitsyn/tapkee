#ifndef libedrt_methods_h_
#define libedrt_methods_h_

struct thread_parameters_t
{
	int N, k, target_dimension, thread, num_threads;
	const int* neighborhood_matrix;
	double (*distance)(int, int, const void*);
	double (*kernel)(int, int, const void*);
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T* pthread_lock;
#endif
	const void* user_data;
	double* W_matrix;
};


// Kernel Locally Linear Embedding

double* klle_weight_matrix(
		int* neighborhood_matrix,
		int N,
		int k,
		int matrix_k,
		int num_threads,
		double reconstruction_shift,
		double (*kernel)(int, int, const void*),
		const void* user_data);

void* klle_weight_matrix_thread(void* params);

// Kernel Local Tangent Space Alignment

double* kltsa_weight_matrix(
		int* neighborhood_matrix,
		int N,
		int k,
		int matrix_k,
		int target_dim,
		int num_threads,
		double (*kernel)(int, int, const void*),
		const void* user_data);

void* kltsa_weight_matrix_thread(void* params);

#endif
