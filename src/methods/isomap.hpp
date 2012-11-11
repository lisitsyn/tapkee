#ifndef TAPKEE_ISOMAP_H_
#define TAPKEE_ISOMAP_H_

#include "../defines.hpp"
#include "../utils/time.hpp"
#include "../utils/fibonacci_heap.hpp"

#include <vector>
#include <limits>
using std::vector;
using std::numeric_limits;

// TODO inplace whether possible
DenseSymmetricMatrix isomap_relax_distances(const DenseSymmetricMatrix& distances, const Neighbors& neighbors)
{
	timed_context context("Distances shortest path relaxing");
	unsigned int n_neighbors = neighbors[0].size();
	unsigned int N = distances.cols();
	FibonacciHeap* heap = new FibonacciHeap(N);

	bool* s = new bool[N];
	bool* f = new bool[N];

	DenseSymmetricMatrix shortest_distances(N,N);
	
	for (unsigned int k=0; k<N; k++)
	{
		// fill s and f with false, fill shortest_D with infinity
		for (unsigned int j=0; j<N; j++)
		{
			shortest_distances(k,j) = numeric_limits<DenseMatrix::Scalar>::max();
			s[j] = false;
			f[j] = false;
		}
		// set distance from k to k as zero
		shortest_distances(k,k) = 0.0;

		// insert kth object to heap with zero distance and set f[k] true
		heap->insert(k,0.0);
		f[k] = true;

		// while heap is not empty
		while (heap->get_num_nodes()>0)
		{
			// extract min and set (s)olution state as true and (f)rontier as false
			double tmp;
			int min_item = heap->extract_min(tmp);
			s[min_item] = true;
			f[min_item] = false;

			// for-each edge (min_item->w)
			for (unsigned int i=0; i<n_neighbors; i++)
			{
				// get w idx
				int w = neighbors[min_item][i];
				// if w is not in solution yet
				if (s[w] == false)
				{
					// get distance from k to i through min_item
					double dist = shortest_distances(k,min_item) + distances(min_item,i);
					// if distance can be relaxed
					if (dist < shortest_distances(k,w))
					{
						// relax distance
						shortest_distances(k,w) = dist;
						// if w is in (f)rontier
						if (f[w])
						{
							// decrease distance in heap
							heap->decrease_key(w, dist);
						}
						else
						{
							// insert w to heap and set (f)rontier as true
							heap->insert(w, dist);
							f[w] = true;
						}
					}
				}
			}
		}
		// clear heap to re-use
		heap->clear();
	}
	delete heap;
	delete[] s;
	delete[] f;
	return shortest_distances;
}

#endif
