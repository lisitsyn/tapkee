#include "libedrt_neighbors.hpp"
#include <pthread.h>
#include <vector>
#include <map>
#include <stdlib.h>
#include <string.h>

#define MAX(x,y) x>y ? x : y

int* neighbors_matrix(
		int N, 
		int k, 
		double (*distance)(int, int, const void*), 
		double (*kernel)(int, int, const void*),
		const void* user_data)
{
	int i,j;
	int* neighborhood_matrix;// = SG_MALLOC(int, N*k);
	double max_dist = 0.0;
//	if (distance && kernel)
//		SG_SERROR("Ambiguous usage");
	if (kernel)
	{
		for (i=0; i<N; i++)
			max_dist = MAX(kernel(i,i,user_data),max_dist);
	}
	else if (distance)
	{
		for (i=0; i<N; i++)
		{
			for (j=i; j<N; j++)
				max_dist = MAX(distance(i,j,user_data),max_dist);
		}
	}
//	else
//		SG_SERROR("Neither kernel nor distance is provided");

//	shogun::CoverTree<covertree_kernel_point_t>* cover_tree = 
//	    new shogun::CoverTree<covertree_kernel_point_t>(max_dist);

//	for (i=0; i<N; i++)
//		cover_tree->insert(covertree_kernel_point_t(i,kernel,user_data));
	
//	for (i=0; i<N; i++)
//	{
//		std::vector<covertree_kernel_point_t> neighbors = 
//		    cover_tree->kNearestNeighbors(
//		        covertree_kernel_point_t(i,kernel,user_data),k+1);
		
//		for (std::size_t m=1; m < unsigned(k+1); m++)
//			neighborhood_matrix[i*k+m-1] = neighbors[m].point_index;
//	}

//	delete cover_tree;

	return neighborhood_matrix;
}

