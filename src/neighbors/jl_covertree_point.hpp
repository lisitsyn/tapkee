/* * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Written (W) John Langford and Dinoj Surendran, v_array and its templatization
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _JL_COVERTREE_POINT_H_
#define _JL_COVERTREE_POINT_H_

#include <iostream>

/** @brief Class v_array taken directly from JL's implementation */
template<class T> 
class v_array{

	public:
		/** Getter for the the last element of the v_array
		 *  @return the last element of the array */
		T last() { return elements[index-1];}

		/** Decrement the pointer to the last element */
		void decr() { index--;}

		/** Create an empty v_array */
		v_array() : index(0), length(0), elements(NULL) {}

		/** Element access operator
		 *  @param i of the element to be read
		 *  @return the corresponding element */
		T& operator[](unsigned int i) { return elements[i]; }

	public:
		/** Pointer to the last element of the v_array */
		int index;

		/** Length of the v_array */
		int length;

		/** Pointer to the beginning of the v_array elements */
		T* elements;

};

/**
 * Insert a new element at the end of the vector
 *
 * @param v vector
 * @param new_ele element to insert
 */
template<class T> 
void push(v_array<T>& v, const T &new_ele)
{
	while(v.index >= v.length)
	{
		v.length = 2*v.length + 3;
		v.elements = (T *)realloc(v.elements,sizeof(T) * v.length);
	}
	v[v.index++] = new_ele;
}

/**
 * Used to modify the capacity of the vector
 *
 * @param v vector
 * @param length the new length of the vector
 */
template<class T> 
void alloc(v_array<T>& v, int length)
{
	v.elements = (T *)realloc(v.elements, sizeof(T) * length);
	v.length = length;
}

/**
 * Returns the vector previous to the pointed one in the stack of
 * vectors and decrements the index of the stack. No memory is
 * freed here. If there are no vectors stored in the stack, create
 * and return a new empty vector
 *
 * @param stack of vectors
 * @return the adequate vector according to the previous conditions
 */
template<class T> 
v_array<T> pop(v_array<v_array<T> > &stack)
{
	if (stack.index > 0)
		return stack[--stack.index];
	else
		return v_array<T>();
}

/** @brief Class Point to use with John Langford's CoverTree. This
 * class must have some associated functions defined (distance,
 * and print, see below) so it can be used with the JLCoverTree
 * implementation.
 */
template <class RandomAccessIterator, class DistanceCallback>
struct JLCoverTreePoint
{
	JLCoverTreePoint() : iter_(NULL), dc_(NULL) {};
	JLCoverTreePoint(const RandomAccessIterator& iter, const DistanceCallback& dc) :
		iter_(iter), dc_(dc) {};

	RandomAccessIterator iter_;
	// TODO not every point needs its own DistanceCallback...
	DistanceCallback dc_;
}; /* struct JLCoverTreePoint */

/** Functions declared out of the class definition to respect JLCoverTree 
 *  structure */

template <class RandomAccessIterator, class DistanceCallback>
double distance(const JLCoverTreePoint<RandomAccessIterator, DistanceCallback>& l,
	        const JLCoverTreePoint<RandomAccessIterator, DistanceCallback>& r, double upper_bound)
{
	assert(upper_bound>=0);
	return l.dc_(*l.iter_,*l.iter_) + l.dc_(*r.iter_,*r.iter_) - 2*l.dc_(*r.iter_,*l.iter_);
}

/** Print the information of the CoverTree point */
template <class RandomAccessIterator, class DistanceCallback>
void print(const JLCoverTreePoint<RandomAccessIterator, DistanceCallback> &p)
{
	std::cout << "Print JLCoverTreePoint not implemented\n";
}

#endif /* _JL_COVERTREE_POINT_H_*/
