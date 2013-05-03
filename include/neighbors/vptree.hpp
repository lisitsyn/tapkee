/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_VPTREE_H_
#define TAPKEE_VPTREE_H_

#include <tapkee_defines.hpp>

#include <vector>
#include <queue>
#include <algorithm>
#include <limits>

namespace tapkee
{
namespace tapkee_internal
{

template<bool, class RandomAccessIterator, class DistanceCallback> 
struct compare_if_kernel;

template<class RandomAccessIterator, class DistanceCallback>
struct DistanceComparator
{
	DistanceCallback callback;
	const RandomAccessIterator item;
	DistanceComparator(const DistanceCallback& c, const RandomAccessIterator& i) :
		callback(c), item(i) {}
	inline bool operator()(const RandomAccessIterator& a, const RandomAccessIterator& b)
	{
		return compare_if_kernel<DistanceCallback::is_kernel,RandomAccessIterator,DistanceCallback>(item,callback)(a,b);
	}
};

template<class RandomAccessIterator, class DistanceCallback> 
struct compare_if_kernel<true,RandomAccessIterator,DistanceCallback>
{
	compare_if_kernel(RandomAccessIterator i, DistanceCallback c) : item(i), callback(c) {}
	bool operator()(const RandomAccessIterator& a, const RandomAccessIterator& b)
	{
		return (-2*callback(item,a) + callback(a,a)) < (-2*callback(item,b) + callback(b,b));
	}
	RandomAccessIterator item;
	DistanceCallback callback;
};

template<class RandomAccessIterator, class DistanceCallback> 
struct compare_if_kernel<false,RandomAccessIterator,DistanceCallback>
{
	compare_if_kernel(RandomAccessIterator i, DistanceCallback c) : item(i), callback(c) {}
	bool operator()(const RandomAccessIterator& a, const RandomAccessIterator& b)
	{
		return callback(item,a) < callback(item,b);
	}
	RandomAccessIterator item;
	DistanceCallback callback;
};

template<class RandomAccessIterator, class DistanceCallback>
class VantagePointTree
{
public:

	// Default constructor
	VantagePointTree(RandomAccessIterator b, RandomAccessIterator e, DistanceCallback c) :  
		begin(b), items(), callback(c), tau(0.0), root(0)
	{
		items.reserve(e-b);
		for (RandomAccessIterator i=b; i!=e; ++i)
			items.push_back(i);
		root = buildFromPoints(0, items.size());
	}

	// Destructor
	~VantagePointTree() 
	{
		delete root;
	}

	// Function that uses the tree to find the k nearest neighbors of target
	std::vector<IndexType> search(const RandomAccessIterator& target, int k)
	{
		std::vector<IndexType> results;
		// Use a priority queue to store intermediate results on
		std::priority_queue<HeapItem> heap;

		// Variable that tracks the distance to the farthest point in our results
		tau = std::numeric_limits<double>::max();

		// Perform the searcg
		search(root, target, k, heap);

		// Gather final results
		results.reserve(k);
		while(!heap.empty()) {
			results.push_back(items[heap.top().index]-begin);
			heap.pop();
		}
		return results;
	}

private:

	VantagePointTree(const VantagePointTree&);
	VantagePointTree& operator=(const VantagePointTree&);

	std::vector<RandomAccessIterator> items;
	RandomAccessIterator begin;
	DistanceCallback callback;
	double tau;

	// Single node of a VP tree (has a point and radius; left children are closer to point than the radius)
	struct Node
	{
		int index;              // index of point in node
		double threshold;       // radius(?)
		Node* left;             // points closer by than threshold
		Node* right;            // points farther away than threshold

		Node() : index(0), threshold(0.), left(0), right(0) {}

		~Node() 
		{
			delete left;
			delete right;
		}

		Node(const Node&);
		Node& operator=(const Node&);

	}* root;

	// An item on the intermediate result queue
	struct HeapItem {
		HeapItem(int indexv, double distv) :
			index(indexv), dist(distv) {}
		int index;
		double dist;
		bool operator<(const HeapItem& o) const {
			return dist < o.dist;
		}
	};


	// Function that (recursively) fills the tree
	Node* buildFromPoints(int lower, int upper)
	{
		if (upper == lower) {     // indicates that we're done here!
			return NULL;
		}

		// Lower index is center of current node
		Node* node = new Node();
		node->index = lower;

		if (upper - lower > 1) {      // if we did not arrive at leaf yet

			// Choose an arbitrary point and move it to the start
			int i = (int) ((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
			std::swap(items[lower], items[i]);

			// Partition around the median distance
			int median = (upper + lower) / 2;
			std::nth_element(items.begin() + lower + 1, items.begin() + median, 
					items.begin() + upper, DistanceComparator<RandomAccessIterator,DistanceCallback>(callback,items[lower]));

			// Threshold of the new node will be the distance to the median
			node->threshold = callback.distance(items[lower], items[median]);

			// Recursively build tree
			node->index = lower;
			node->left = buildFromPoints(lower + 1, median);
			node->right = buildFromPoints(median, upper);
		}

		// Return result
		return node;
	}

	// Helper function that searches the tree    
	void search(Node* node, const RandomAccessIterator& target, int k, std::priority_queue<HeapItem>& heap)
	{
		if(node == NULL) return;     // indicates that we're done here

		// Compute distance between target and current node
		double dist = callback.distance(items[node->index], target);

		// If current node within radius tau
		if(dist < tau) {
			if(heap.size() == static_cast<size_t>(k)) heap.pop(); // remove furthest node from result list (if we already have k results)
			heap.push(HeapItem(node->index, dist));           // add current node to result list
			if(heap.size() == static_cast<size_t>(k)) tau = heap.top().dist;     // update value of tau (farthest point in result list)
		}

		// Return if we arrived at a leaf
		if(node->left == NULL && node->right == NULL) {
			return;
		}

		// If the target lies within the radius of ball
		if(dist < node->threshold) {
			if(dist - tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child first
				search(node->left, target, k, heap);
			}

			if(dist + tau >= node->threshold) {         // if there can still be neighbors outside the ball, recursively search right child
				search(node->right, target, k, heap);
			}

			// If the target lies outsize the radius of the ball
		} else {
			if(dist + tau >= node->threshold) {         // if there can still be neighbors outside the ball, recursively search right child first
				search(node->right, target, k, heap);
			}

			if (dist - tau <= node->threshold) {         // if there can still be neighbors inside the ball, recursively search left child
				search(node->left, target, k, heap);
			}
		}
	}
};

}
}
#endif
