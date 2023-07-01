/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2009-2013 John Langford, Dinoj Surendran, Fernando José Iglesias García
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/neighbors/covertree_point.hpp>
/* End of Tapkee includes */

/* First written by John Langford jl@hunch.net
   Templatization by Dinoj Surendran dinojs@gmail.com
   Adaptation to Shogun by Fernando José Iglesias García
 */
namespace tapkee
{
namespace tapkee_internal
{

/**
 * Cover tree node TODO better doc
 */
template <class P> struct node
{
    node() : p(), max_dist(0.0), parent_dist(0.0), children(), num_children(0), scale(0)
    {
    }

    node(P _p, ScalarType _max_dist, ScalarType _parent_dist, std::vector<node<P>> _children, unsigned short int _num_children,
         short int _scale)
        : p(_p), max_dist(_max_dist), parent_dist(_parent_dist), children(_children), num_children(_num_children),
          scale(_scale)
    {
    }

    /** Point */
    P p;

    /** The maximum distance to any grandchild */
    ScalarType max_dist;

    /** The distance to the parent */
    ScalarType parent_dist;

    /** Pointer to the list of children of this node */
    std::vector<node<P>> children;

    /** The number of children nodes of this node */
    unsigned short int num_children;

    /** Essentially, an upper bound on the distance to any child */
    short int scale;
};

/**
 * Cover tree node with an associated set of distances TODO better doc
 */
template <class P> struct ds_node
{

    ds_node() : dist(), p()
    {
    }

    /** Vector of distances TODO better doc*/
    v_array<ScalarType> dist;

    /** Point TODO better doc */
    P p;
};

/**
 * List of cover tree nodes associated to a distance TODO better doc
 */
template <class P> struct d_node
{
    /** Distance TODO better doc*/
    ScalarType dist;

    /** List of nodes TODO better doc*/
    const node<P>* n;
};

template <class P> inline ScalarType compare(const d_node<P>& p1, const d_node<P>& p2)
{
    return p1.dist - p2.dist;
}

template <class P> node<P> new_node(const P& p)
{
    node<P> new_node;
    new_node.p = p;
    return new_node;
}

template <class P> node<P> new_leaf(const P& p)
{
    node<P> new_leaf(p, 0., 0., {}, 0, 100);
    return new_leaf;
}

}
}
