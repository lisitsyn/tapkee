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
#include <tapkee/neighbors/covertree/structures.hpp>
#include <tapkee/neighbors/covertree_point.hpp>
/* End of Tapkee includes */

#include <assert.h>
#include <cmath>
#include <limits>
#include <stdio.h>

/* First written by John Langford jl@hunch.net
   Templatization by Dinoj Surendran dinojs@gmail.com
   Adaptation to Shogun by Fernando José Iglesias García
 */
namespace tapkee
{
namespace tapkee_internal
{

template <class P, class DistanceCallback> class CoverTreeWrapper
{
  public:
    CoverTreeWrapper() : base(COVERTREE_BASE), il2(1. / log(base)), internal_k(1)
    {
    }

    void split(v_array<ds_node<P>>& point_set, v_array<ds_node<P>>& far_set, int max_scale);

    void dist_split(DistanceCallback& dcb, v_array<ds_node<P>>& point_set, v_array<ds_node<P>>& new_point_set,
                    P new_point, int max_scale);

    node<P> batch_insert(DistanceCallback& dcb, const P& p, int max_scale, int top_scale,
                         v_array<ds_node<P>>& point_set, v_array<ds_node<P>>& consumed_set,
                         v_array<v_array<ds_node<P>>>& stack);

    node<P> batch_create(DistanceCallback& dcb, v_array<P> points);

    void k_nearest_neighbor(DistanceCallback& dcb, const node<P>& top_node, const node<P>& query,
                            v_array<v_array<P>>& results, int k);

    void batch_nearest_neighbor(DistanceCallback& dcb, const node<P>& top_node, const node<P>& query,
                                v_array<v_array<P>>& results);

    void internal_batch_nearest_neighbor(DistanceCallback& dcb, const node<P>* query,
                                         v_array<v_array<d_node<P>>>& cover_sets, v_array<d_node<P>>& zero_set,
                                         int current_scale, int max_scale, std::vector<ScalarType>& upper_bound,
                                         v_array<v_array<P>>& results,
                                         v_array<v_array<v_array<d_node<P>>>>& spare_cover_sets,
                                         v_array<v_array<d_node<P>>>& spare_zero_sets);

    void brute_nearest(DistanceCallback& dcb, const node<P>* query, v_array<d_node<P>> zero_set,
                       std::vector<ScalarType>& upper_bound, v_array<v_array<P>>& results,
                       v_array<v_array<d_node<P>>>& spare_zero_sets);

    void descend(DistanceCallback& dcb, const node<P>* query, std::vector<ScalarType>& upper_bound, int current_scale,
                 int& max_scale, v_array<v_array<d_node<P>>>& cover_sets, v_array<d_node<P>>& zero_set);

    void copy_cover_sets(DistanceCallback& dcb, const node<P>* query_chi, std::vector<ScalarType>& new_upper_bound,
                         v_array<v_array<d_node<P>>>& cover_sets, v_array<v_array<d_node<P>>>& new_cover_sets,
                         int current_scale, int max_scale);

    void copy_zero_set(DistanceCallback& dcb, const node<P>* query_chi, std::vector<ScalarType>& new_upper_bound,
                       v_array<d_node<P>>& zero_set, v_array<d_node<P>>& new_zero_set);

    v_array<v_array<d_node<P>>> get_cover_sets(v_array<v_array<v_array<d_node<P>>>>& spare_cover_sets);

    inline ScalarType dist_of_scale(int s)
    {
        return pow(base, s);
    }

    inline int get_scale(ScalarType d)
    {
        return (int)ceil(il2 * log(d));
    }

    void update(std::vector<ScalarType>& k_upper_bound, ScalarType upper_bound)
    {
        auto end = k_upper_bound.begin() + internal_k - 1;
        auto begin = k_upper_bound.begin();
        for (; end != begin; begin++)
        {
            if (upper_bound < *(begin + 1))
                *begin = *(begin + 1);
            else
            {
                *begin = upper_bound;
                break;
            }
        }
        if (end == begin)
            *begin = upper_bound;
    }

    void setter(std::vector<ScalarType>& vector, ScalarType max)
    {
        auto begin = vector.begin();
        for (auto end = begin + internal_k; end != begin; begin++)
            *begin = max;
    }

    std::vector<ScalarType> alloc_upper()
    {
        return std::vector<ScalarType>(internal_k);
    }

  private:
    ScalarType base;
    ScalarType il2;
    int internal_k;
};

template <class P> ScalarType max_set(v_array<ds_node<P>>& v)
{
    ScalarType max = 0.;
    for (int i = 0; i < size(v); i++)
        if (max < v[i].dist.last())
            max = v[i].dist.last();
    return max;
}

template <class P, class D>
void CoverTreeWrapper<P, D>::split(v_array<ds_node<P>>& point_set, v_array<ds_node<P>>& far_set, int max_scale)
{
    IndexType new_index = 0;
    ScalarType fmax = dist_of_scale(max_scale);
    for (int i = 0; i < size(point_set); i++)
    {
        if (point_set[i].dist.last() <= fmax)
        {
            point_set[new_index++] = point_set[i];
        }
        else
            push(far_set, point_set[i]);
    }
    resize(point_set, new_index);
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::dist_split(DistanceCallback& dcb, v_array<ds_node<P>>& point_set,
                                                       v_array<ds_node<P>>& new_point_set, P new_point, int max_scale)
{
    IndexType new_index = 0;
    ScalarType fmax = dist_of_scale(max_scale);
    for (int i = 0; i < size(point_set); i++)
    {
        ScalarType new_d;
        new_d = distance(dcb, new_point, point_set[i].p, fmax);
        if (new_d <= fmax)
        {
            push(point_set[i].dist, new_d);
            push(new_point_set, point_set[i]);
        }
        else
            point_set[new_index++] = point_set[i];
    }
    resize(point_set, new_index);
}

/*
   max_scale is the maximum scale of the node we might create here.
   point_set contains points which are 2*max_scale or less away.
   */
template <class P, class DistanceCallback>
node<P> CoverTreeWrapper<P, DistanceCallback>::batch_insert(DistanceCallback& dcb, const P& p, int max_scale,
                                                            int top_scale, v_array<ds_node<P>>& point_set,
                                                            v_array<ds_node<P>>& consumed_set,
                                                            v_array<v_array<ds_node<P>>>& stack)
{
    if (size(point_set) == 0)
        return new_leaf(p);
    else
    {
        ScalarType max_dist = max_set(point_set); // O(|point_set|)
        int next_scale = std::min(max_scale - 1, get_scale(max_dist));
        if (next_scale == -2147483647 - 1) // We have points with distance 0.
        {
            v_array<node<P>> children;
            push(children, new_leaf(p));
            while (size(point_set) > 0)
            {
                push(children, new_leaf(point_set.last().p));
                push(consumed_set, point_set.last());
                point_set.decr();
            }
            node<P> n = new_node(p);
            n.scale = 100; // A magic number meant to be larger than all scales.
            n.max_dist = 0;
            alloc_array(children, size(children));
            n.num_children = size(children);
            n.children = children.elements;
            return n;
        }
        else
        {
            v_array<ds_node<P>> far = pop(stack);
            split(point_set, far, max_scale); // O(|point_set|)

            node<P> child = batch_insert(dcb, p, next_scale, top_scale, point_set, consumed_set, stack);

            if (size(point_set) == 0)
            {
                push(stack, point_set);
                point_set = far;
                return child;
            }
            else
            {
                node<P> n = new_node(p);
                v_array<node<P>> children;
                push(children, child);
                v_array<ds_node<P>> new_point_set = pop(stack);
                v_array<ds_node<P>> new_consumed_set = pop(stack);
                while (size(point_set) != 0)
                { // O(|point_set| * num_children)
                    P new_point = point_set.last().p;
                    ScalarType new_dist = point_set.last().dist.last();
                    push(consumed_set, point_set.last());
                    point_set.decr();

                    dist_split(dcb, point_set, new_point_set, new_point, max_scale); // O(|point_saet|)
                    dist_split(dcb, far, new_point_set, new_point, max_scale);       // O(|far|)

                    node<P> new_child =
                        batch_insert(dcb, new_point, next_scale, top_scale, new_point_set, new_consumed_set, stack);
                    new_child.parent_dist = new_dist;

                    push(children, new_child);

                    ScalarType fmax = dist_of_scale(max_scale);
                    for (int i = 0; i < size(new_point_set); i++) // O(|new_point_set|)
                    {
                        new_point_set[i].dist.decr();
                        if (new_point_set[i].dist.last() <= fmax)
                            push(point_set, new_point_set[i]);
                        else
                            push(far, new_point_set[i]);
                    }
                    for (int i = 0; i < size(new_consumed_set); i++) // O(|new_point_set|)
                    {
                        new_consumed_set[i].dist.decr();
                        push(consumed_set, new_consumed_set[i]);
                    }
                    resize(new_point_set, 0);
                    resize(new_consumed_set, 0);
                }
                push(stack, new_point_set);
                push(stack, new_consumed_set);
                push(stack, point_set);
                point_set = far;
                n.scale = top_scale - max_scale;
                n.max_dist = max_set(consumed_set);
                alloc_array(children, size(children));
                n.num_children = size(children);
                n.children = children.elements;
                return n;
            }
        }
    }
}

template <class P, class DistanceCallback>
node<P> CoverTreeWrapper<P, DistanceCallback>::batch_create(DistanceCallback& dcb, v_array<P> points)
{
    assert(size(points) > 0);
    v_array<ds_node<P>> point_set;
    v_array<v_array<ds_node<P>>> stack;

    for (int i = 1; i < size(points); i++)
    {
        ds_node<P> temp;
        push(temp.dist, distance(dcb, points[0], points[i], std::numeric_limits<ScalarType>::max()));
        temp.p = points[i];
        push(point_set, temp);
    }

    v_array<ds_node<P>> consumed_set;

    ScalarType max_dist = max_set(point_set);

    node<P> top =
        batch_insert(dcb, points[0], get_scale(max_dist), get_scale(max_dist), point_set, consumed_set, stack);
    return top;
}

void add_height(int d, v_array<int>& heights)
{
    if (size(heights) <= d)
        for (; size(heights) <= d;)
            push(heights, 0);
    heights[d] = heights[d] + 1;
}

template <class P> int height_dist(const node<P> top_node, v_array<int>& heights)
{
    if (top_node.num_children == 0)
    {
        add_height(0, heights);
        return 0;
    }
    else
    {
        int max_v = 0;
        for (int i = 0; i < top_node.num_children; i++)
        {
            int d = height_dist(top_node.children[i], heights);
            if (d > max_v)
                max_v = d;
        }
        add_height(1 + max_v, heights);
        return (1 + max_v);
    }
}

template <class P> void depth_dist(int top_scale, const node<P> top_node, v_array<int>& depths)
{
    if (top_node.num_children > 0)
        for (int i = 0; i < top_node.num_children; i++)
        {
            add_height(top_node.scale, depths);
            depth_dist(top_scale, top_node.children[i], depths);
        }
}

template <class P> void breadth_dist(const node<P> top_node, v_array<int>& breadths)
{
    if (top_node.num_children == 0)
        add_height(0, breadths);
    else
    {
        for (int i = 0; i < top_node.num_children; i++)
            breadth_dist(top_node.children[i], breadths);
        add_height(top_node.num_children, breadths);
    }
}

template <class P> void halfsort(v_array<d_node<P>> cover_set)
{
    if (size(cover_set) <= 1)
        return;
    auto base_ptr = begin(cover_set);

    auto hi = base_ptr + size(cover_set) - 1;
    auto right_ptr = hi;
    auto left_ptr = base_ptr;

    while (right_ptr > base_ptr)
    {
        auto mid = base_ptr + ((hi - base_ptr) >> 1);

        if (compare(*mid, *base_ptr) < 0.)
            std::swap(*mid, *base_ptr);
        if (compare(*hi, *mid) < 0.)
            std::swap(*mid, *hi);
        else
            goto jump_over;
        if (compare(*mid, *base_ptr) < 0.)
            std::swap(*mid, *base_ptr);
    jump_over:;

        left_ptr = base_ptr + 1;
        right_ptr = hi - 1;

        do
        {
            while (compare(*left_ptr, *mid) < 0.)
                left_ptr++;

            while (compare(*mid, *right_ptr) < 0.)
                right_ptr--;

            if (left_ptr < right_ptr)
            {
                std::swap(*left_ptr, *right_ptr);
                if (mid == left_ptr)
                    mid = right_ptr;
                else if (mid == right_ptr)
                    mid = left_ptr;
                left_ptr++;
                right_ptr--;
            }
            else if (left_ptr == right_ptr)
            {
                left_ptr++;
                right_ptr--;
                break;
            }
        } while (left_ptr <= right_ptr);
        hi = right_ptr;
    }
}

template <class P, class D>
v_array<v_array<d_node<P>>> CoverTreeWrapper<P, D>::get_cover_sets(
    v_array<v_array<v_array<d_node<P>>>>& spare_cover_sets)
{
    v_array<v_array<d_node<P>>> ret = pop(spare_cover_sets);
    while (size(ret) < 101)
    {
        v_array<d_node<P>> temp;
        push(ret, temp);
    }
    return ret;
}

inline bool shell(ScalarType parent_query_dist, ScalarType child_parent_dist, ScalarType upper_bound)
{
    return parent_query_dist - child_parent_dist <= upper_bound;
    //    && child_parent_dist - parent_query_dist <= upper_bound;
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::copy_zero_set(DistanceCallback& dcb, const node<P>* query_chi,
                                                          std::vector<ScalarType>& new_upper_bound,
                                                          v_array<d_node<P>>& zero_set,
                                                          v_array<d_node<P>>& new_zero_set)
{
    resize(new_zero_set, 0);
    auto end = begin(zero_set) + size(zero_set);
    for (auto ele = begin(zero_set); ele != end; ele++)
    {
        ScalarType upper_dist = new_upper_bound[0] + query_chi->max_dist;
        if (shell(ele->dist, query_chi->parent_dist, upper_dist))
        {
            ScalarType d = distance(dcb, query_chi->p, ele->n->p, upper_dist);

            if (d <= upper_dist)
            {
                if (d < new_upper_bound[0])
                    update(new_upper_bound, d);
                d_node<P> temp = {d, ele->n};
                push(new_zero_set, temp);
            }
        }
    }
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::copy_cover_sets(DistanceCallback& dcb, const node<P>* query_chi,
                                                            std::vector<ScalarType>& new_upper_bound,
                                                            v_array<v_array<d_node<P>>>& cover_sets,
                                                            v_array<v_array<d_node<P>>>& new_cover_sets,
                                                            int current_scale, int max_scale)
{
    for (; current_scale <= max_scale; current_scale++)
    {
        auto ele = begin(cover_sets[current_scale]);
        auto end = begin(cover_sets[current_scale]) + size(cover_sets[current_scale]);
        for (; ele != end; ele++)
        {
            ScalarType upper_dist = new_upper_bound[0] + query_chi->max_dist + ele->n->max_dist;
            if (shell(ele->dist, query_chi->parent_dist, upper_dist))
            {
                ScalarType d = distance(dcb, query_chi->p, ele->n->p, upper_dist);

                if (d <= upper_dist)
                {
                    if (d < new_upper_bound[0])
                        update(new_upper_bound, d);
                    d_node<P> temp = {d, ele->n};
                    push(new_cover_sets[current_scale], temp);
                }
            }
        }
    }
}

/*
   An optimization to consider:
   Make all distance evaluations occur in descend.

   Instead of passing a cover_set, pass a stack of cover sets.  The
   last element holds d_nodes with your distance.  The next lower
   element holds a d_node with the distance to your query parent,
   next = query grand parent, etc..

   Compute distances in the presence of the tighter upper bound.
   */
template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::descend(DistanceCallback& dcb, const node<P>* query,
                                                    std::vector<ScalarType>& upper_bound, int current_scale,
                                                    int& max_scale, v_array<v_array<d_node<P>>>& cover_sets,
                                                    v_array<d_node<P>>& zero_set)
{
    auto end = begin(cover_sets[current_scale]) + size(cover_sets[current_scale]);
    for (auto parent = begin(cover_sets[current_scale]); parent != end; parent++)
    {
        const node<P>* par = parent->n;
        ScalarType upper_dist = upper_bound[0] + query->max_dist + query->max_dist;
        if (parent->dist <= upper_dist + par->max_dist)
        {
            auto chi = par->children.begin();
            if (parent->dist <= upper_dist + chi->max_dist)
            {
                if (chi->num_children > 0)
                {
                    if (max_scale < chi->scale)
                        max_scale = chi->scale;
                    d_node<P> temp = {parent->dist, &(*chi)};
                    push(cover_sets[chi->scale], temp);
                }
                else if (parent->dist <= upper_dist)
                {
                    d_node<P> temp = {parent->dist, &(*chi)};
                    push(zero_set, temp);
                }
            }
            auto child_end = par->children.begin() + par->num_children;
            for (chi++; chi != child_end; chi++)
            {
                ScalarType upper_chi = upper_bound[0] + chi->max_dist + query->max_dist + query->max_dist;
                if (shell(parent->dist, chi->parent_dist, upper_chi))
                {
                    ScalarType d = distance(dcb, query->p, chi->p, upper_chi);
                    if (d <= upper_chi)
                    {
                        if (d < upper_bound[0])
                            update(upper_bound, d);
                        if (chi->num_children > 0)
                        {
                            if (max_scale < chi->scale)
                                max_scale = chi->scale;
                            d_node<P> temp = {d, &(*chi)};
                            push(cover_sets[chi->scale], temp);
                        }
                        else if (d <= upper_chi - chi->max_dist)
                        {
                            d_node<P> temp = {d, &(*chi)};
                            push(zero_set, temp);
                        }
                    }
                }
            }
        }
    }
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::brute_nearest(DistanceCallback& dcb, const node<P>* query,
                                                          v_array<d_node<P>> zero_set,
                                                          std::vector<ScalarType>& upper_bound,
                                                          v_array<v_array<P>>& results,
                                                          v_array<v_array<d_node<P>>>& spare_zero_sets)
{
    if (query->num_children > 0)
    {
        v_array<d_node<P>> new_zero_set = pop(spare_zero_sets);
        auto query_chi = query->children.begin();
        brute_nearest(dcb, &(*query_chi), zero_set, upper_bound, results, spare_zero_sets);
        std::vector<ScalarType> new_upper_bound = alloc_upper();

        auto child_end = query->children.begin() + query->num_children;
        for (query_chi++; query_chi != child_end; query_chi++)
        {
            setter(new_upper_bound, upper_bound[0] + query_chi->parent_dist);
            copy_zero_set(dcb, &(*query_chi), new_upper_bound, zero_set, new_zero_set);
            brute_nearest(dcb, &(*query_chi), new_zero_set, new_upper_bound, results, spare_zero_sets);
        }
        resize(new_zero_set, 0);
        push(spare_zero_sets, new_zero_set);
    }
    else
    {
        v_array<P> temp;
        push(temp, query->p);
        auto end = begin(zero_set) + size(zero_set);
        for (auto ele = begin(zero_set); ele != end; ele++)
            if (ele->dist <= upper_bound[0])
                push(temp, ele->n->p);
        push(results, temp);
    }
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::internal_batch_nearest_neighbor(
    DistanceCallback& dcb, const node<P>* query, v_array<v_array<d_node<P>>>& cover_sets, v_array<d_node<P>>& zero_set,
    int current_scale, int max_scale, std::vector<ScalarType>& upper_bound, v_array<v_array<P>>& results,
    v_array<v_array<v_array<d_node<P>>>>& spare_cover_sets, v_array<v_array<d_node<P>>>& spare_zero_sets)
{
    if (current_scale > max_scale) // All remaining points are in the zero set.
        brute_nearest(dcb, query, zero_set, upper_bound, results, spare_zero_sets);
    else if (query->scale <= current_scale && query->scale != 100)
    // Our query has too much scale.  Reduce.
    {
        auto query_chi = query->children.begin();
        v_array<d_node<P>> new_zero_set = pop(spare_zero_sets);
        v_array<v_array<d_node<P>>> new_cover_sets = get_cover_sets(spare_cover_sets);
        std::vector<ScalarType> new_upper_bound = alloc_upper();

        auto child_end = query->children.begin() + query->num_children;
        for (query_chi++; query_chi != child_end; query_chi++)
        {
            setter(new_upper_bound, upper_bound[0] + query_chi->parent_dist);
            copy_zero_set(dcb, &(*query_chi), new_upper_bound, zero_set, new_zero_set);
            copy_cover_sets(dcb, &(*query_chi), new_upper_bound, cover_sets, new_cover_sets, current_scale, max_scale);
            internal_batch_nearest_neighbor(dcb, &(*query_chi), new_cover_sets, new_zero_set, current_scale, max_scale,
                                            new_upper_bound, results, spare_cover_sets, spare_zero_sets);
        }
        resize(new_zero_set, 0);
        push(spare_zero_sets, new_zero_set);
        push(spare_cover_sets, new_cover_sets);
        internal_batch_nearest_neighbor(dcb, &query->children[0], cover_sets, zero_set, current_scale, max_scale,
                                        upper_bound, results, spare_cover_sets, spare_zero_sets);
    }
    else // reduce cover set scale
    {
        halfsort(cover_sets[current_scale]);
        descend(dcb, query, upper_bound, current_scale, max_scale, cover_sets, zero_set);
        resize(cover_sets[current_scale++], 0);
        internal_batch_nearest_neighbor(dcb, query, cover_sets, zero_set, current_scale, max_scale, upper_bound,
                                        results, spare_cover_sets, spare_zero_sets);
    }
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::batch_nearest_neighbor(DistanceCallback& dcb, const node<P>& top_node,
                                                                   const node<P>& query, v_array<v_array<P>>& results)
{
    v_array<v_array<v_array<d_node<P>>>> spare_cover_sets;
    v_array<v_array<d_node<P>>> spare_zero_sets;

    v_array<v_array<d_node<P>>> cover_sets = get_cover_sets(spare_cover_sets);
    v_array<d_node<P>> zero_set = pop(spare_zero_sets);

    std::vector<ScalarType> upper_bound = alloc_upper();
    setter(upper_bound, std::numeric_limits<ScalarType>::max());

    ScalarType top_dist = distance(dcb, query.p, top_node.p, std::numeric_limits<ScalarType>::max());
    update(upper_bound, top_dist);

    d_node<P> temp = {top_dist, &top_node};
    push(cover_sets[0], temp);

    internal_batch_nearest_neighbor(dcb, &query, cover_sets, zero_set, 0, 0, upper_bound, results, spare_cover_sets,
                                    spare_zero_sets);

    push(spare_cover_sets, cover_sets);

    for (int i = 0; i < size(spare_cover_sets); i++)
    {
        v_array<v_array<d_node<P>>> cover_sets2 = spare_cover_sets[i];
    }

    push(spare_zero_sets, zero_set);
}

template <class P, class DistanceCallback>
void CoverTreeWrapper<P, DistanceCallback>::k_nearest_neighbor(DistanceCallback& dcb, const node<P>& top_node,
                                                               const node<P>& query, v_array<v_array<P>>& results,
                                                               int k)
{
    internal_k = k;
    batch_nearest_neighbor(dcb, top_node, query, results);
}

} // namespace tapkee_internal
} // namespace tapkee
