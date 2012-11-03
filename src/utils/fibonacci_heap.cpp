/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Evgeniy Andreev (gsomix)
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "fibonacci_heap.hpp"
#include <cmath>


FibonacciHeap::FibonacciHeap(int capacity) : min_root(NULL), nodes(NULL), num_nodes(0),
	num_trees(0), max_num_nodes(capacity), A(NULL), Dn(0)
{
	nodes = (FibonacciHeapNode**)malloc(sizeof(FibonacciHeapNode*)*max_num_nodes);
	for (int i = 0; i < max_num_nodes; i++)
		nodes[i] = new FibonacciHeapNode;

	Dn = 1 + (int)(log2(double(max_num_nodes)));
	A = (FibonacciHeapNode**)malloc(sizeof(FibonacciHeapNode*)*Dn);
	for (int i = 0; i < Dn; i++)
		A[i] = NULL;

	num_nodes = 0;
	num_trees = 0;
}

FibonacciHeap::~FibonacciHeap()
{
	for(int i = 0; i < max_num_nodes; i++)
	{
		if(nodes[i] != NULL)
			delete nodes[i];
	}
	free(nodes);
	free(A);
}

void FibonacciHeap::insert(int index, double key)
{
	if(index >= static_cast<int>(max_num_nodes) || index < 0)
		return;

	if(nodes[index]->index != -1)
		return; // node is not empty

	// init "new" node in array
	nodes[index]->child = NULL;
	nodes[index]->parent = NULL;

	nodes[index]->rank = 0;
	nodes[index]->index = index;
	nodes[index]->key = key;
	nodes[index]->marked = false;

	add_to_roots(nodes[index]);
	num_nodes++;
}

int FibonacciHeap::extract_min(double& ret_key)
{
	FibonacciHeapNode *min_node;
	FibonacciHeapNode *child, *next_child;

	int result;

	if(num_nodes == 0)
		return -1;

	min_node = min_root;
	if(min_node == NULL)
		return -1; // heap is empty now

	child = min_node->child;
	while(child != NULL && child->parent != NULL)
	{
		next_child = child->right;

		// delete current child from childs list
		child->left->right = child->right;
		child->right->left = child->left;

		add_to_roots(child);

		// next iteration
		child = next_child;
	}

	// delete minimun from root list
	min_node->left->right = min_node->right;
	min_node->right->left = min_node->left;

	num_trees--;

	if(min_node == min_node->right)
	{
		min_root = NULL; // remove last element
	}
	else
	{
		min_root = min_node->right;
		consolidate();
	}

	result = min_node->index;
	ret_key = min_node->key;
	clear_node(result);

	num_nodes--;

	return result;
}

void FibonacciHeap::clear()
{
	min_root = NULL;

	// clear all nodes
	for(int i = 0; i < max_num_nodes; i++)
	{
		clear_node(i);
	}

	num_nodes = 0;
	num_trees = 0;
}

int FibonacciHeap::get_key(int index, double& ret_key)
{
	if(index >= max_num_nodes || index < 0)
		return -1;
	if(nodes[index]->index == -1)
		return -1;

	int result = nodes[index]->index;
	ret_key = nodes[index]->key;

	return result;
}

void FibonacciHeap::decrease_key(int index, double& key)
{
	FibonacciHeapNode* parent;

	if(index >= max_num_nodes || index < 0)
		return;
	if(nodes[index]->index == -1)
		return; // node is empty
	if(key > nodes[index]->key)
		return;


	nodes[index]->key = key;

	parent = nodes[index]->parent;

	if(parent != NULL && nodes[index]->key < parent->key)
	{
		cut(nodes[index], parent);
		cascading_cut(parent);
	}

	if(nodes[index]->key < min_root->key)
		min_root = nodes[index];
}


void FibonacciHeap::add_to_roots(FibonacciHeapNode *up_node)
{
	if(min_root == NULL)
	{
		// if heap is empty, node becomes circular root list
		min_root = up_node;

		up_node->left = up_node;
		up_node->right = up_node;
	}
	else
	{
		// insert node to root list
		up_node->right = min_root->right;
		up_node->left = min_root;

		up_node->left->right = up_node;
		up_node->right->left = up_node;

		// nomination of new minimum node
		if(up_node->key < min_root->key)
		{
			min_root = up_node;
		}
	}

	up_node->parent = NULL;
	num_trees++;
}

void FibonacciHeap::consolidate()
{
	FibonacciHeapNode *x, *y, *w;
	int d;

	for(int i = 0; i < Dn; i++)
	{
		A[i] = NULL;
	}

	min_root->left->right = NULL;
	min_root->left = NULL;
	w = min_root;

	do
	{
		x = w;
		d = x->rank;
		w = w->right;

		while(A[d] != NULL)
		{
			y = A[d];

			if(y->key < x->key)
			{
				FibonacciHeapNode *temp;

				temp = y;
				y = x;
				x = temp;
			}

			link_nodes(y, x);

			A[d] = NULL;
			d++;
		}
		A[d] = x;
	}
	while(w != NULL);

	min_root = NULL;
	num_trees = 0;

	for(int i = 0; i < Dn; i++)
	{
		if(A[i] != NULL)
		{
			A[i]->marked = false;
			add_to_roots(A[i]);
		}
	}
}

void FibonacciHeap::link_nodes(FibonacciHeapNode *y, FibonacciHeapNode *x)
{
	if(y->right != NULL)
		y->right->left = y->left;
	if(y->left != NULL)
		y->left->right = y->right;

	num_trees--;

	y->left = y;
	y->right = y;

	y->parent = x;

	if(x->child == NULL)
	{
		x->child = y;
	}
	else
	{
		y->left = x->child;
		y->right = x->child->right;

		x->child->right = y;
		y->right->left = y;
	}

	x->rank++;

	y->marked = false;
}

void FibonacciHeap::cut(FibonacciHeapNode *child, FibonacciHeapNode *parent)
{
	if(parent->child == child)
		parent->child = child->right;

	if(parent->child == child)
		parent->child = NULL;

	parent->rank--;

	child->left->right = child->right;
	child->right->left = child->left;
	child->marked = false;

	add_to_roots(child);
}

void FibonacciHeap::clear_node(int32_t index)
{
	nodes[index]->parent = NULL;
	nodes[index]->child = NULL;
	nodes[index]->left = NULL;
	nodes[index]->right = NULL;

	nodes[index]->rank = 0;
	nodes[index]->index = -1;
	nodes[index]->key = 0;

	nodes[index]->marked = false;
}

void FibonacciHeap::cascading_cut(FibonacciHeapNode *tree)
{
	FibonacciHeapNode *temp;

	temp = tree->parent;
	if(temp != NULL)
	{
		if(!tree->marked)
		{
			tree->marked = true;
		}
		else
		{
			cut(tree, temp);
			cascading_cut(temp);
		}
	}
}
