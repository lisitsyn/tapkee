/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Evgeniy Andreev (gsomix)
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef FIBONACCI_H_
#define FIBONACCI_H_

#include <stdlib.h>
#include <utility>

using std::pair;
using std::make_pair;

struct FibonacciHeapNode
{
	FibonacciHeapNode() : parent(NULL), child(NULL), left(NULL), right(NULL),
		rank(0), marked(false), index(-1), key(0.0)
	{
	}

	/** pointer to parent node */
	FibonacciHeapNode* parent;

	/** pointer to child node */
	FibonacciHeapNode* child;

	/** pointer to left sibling */
	FibonacciHeapNode* left;

	/** pointer to right sibling */
	FibonacciHeapNode* right;

	/** rank of node */
	int rank;

	/** marked flag */
	bool marked;

	/** index in heap */
	int index;

	/** key of node */
	double key;

private:
	FibonacciHeapNode(const FibonacciHeapNode& fh);
	FibonacciHeapNode& operator=(const FibonacciHeapNode& fh);
};

/** @brief the class FibonacciHeap, a fibonacci
 * heap. Generally used by Isomap for Dijkstra heap
 * algorithm
 *
 * w: http://en.wikipedia.org/wiki/Fibonacci_heap
 */
class FibonacciHeap
{
public:

	/** Constructor for heap with specified capacity */
	FibonacciHeap(int capacity);

	virtual ~FibonacciHeap();

	int get_num_nodes() const
	{
		return num_nodes;
	}

	int get_num_trees()
	{
		return num_trees;
	}

	int get_capacity()
	{
		return max_num_nodes;
	}

	/** Inserts nodes with certain key in array of nodes with index
	 * Have time of O(1)
	 */
	void insert(int index, double key);

	/** Deletes and returns item with minimal key
	 * Have amortized time of O(log n)
	 * @return item with minimal key
	 */
	int extract_min(double& ret_val);

	/** Clears all nodes in heap */
	void clear();

	/** Returns key by index
	 * @return -1 if not valid
	 */
	int get_key(int index, double& ret_key);

	/** Decreases key by index
	 * Have amortized time of O(1)
	 */
	void decrease_key(int index, double& key);

private:

	FibonacciHeap();
	FibonacciHeap(const FibonacciHeap& fh);
	FibonacciHeap& operator=(const FibonacciHeap& fh);

private:
	/** Adds node to roots list */
	void add_to_roots(FibonacciHeapNode *up_node);

	/** Consolidates heap */
	void consolidate();

	/** Links right node to childs of left node */
	void link_nodes(FibonacciHeapNode *right, FibonacciHeapNode *left);

	/** Clears node by index */
	void clear_node(int index);

	/** Cuts child node from childs list of parent */
	void cut(FibonacciHeapNode *child, FibonacciHeapNode *parent);

	void cascading_cut(FibonacciHeapNode* tree);

protected:
	/** minimal root in heap */
	FibonacciHeapNode* min_root;

	/** array of nodes for fast search by index */
	FibonacciHeapNode** nodes;

	/** number of nodes */
	int num_nodes;

	/** number of trees */
	int num_trees;

	/** maximum number of nodes */
	int max_num_nodes;

	/** supporting array */
	FibonacciHeapNode **A;

	/** size of supporting array */
	int Dn;
};
#endif /* FIBONACCI_H_ */
