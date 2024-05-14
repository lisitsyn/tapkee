/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando J. Iglesias Garcia
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
#include <tapkee/utils/logging.hpp>
#include <tapkee/utils/naming.hpp>
#ifdef TAPKEE_USE_LGPL_COVERTREE
#include <tapkee/neighbors/covertree.hpp>
#endif
#include <tapkee/neighbors/connected.hpp>
#include <tapkee/neighbors/vptree.hpp>
/* End of Tapkee includes */

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace tapkee
{
namespace tapkee_internal
{

template <class DistanceRecord> struct distances_comparator
{
    inline bool operator()(const DistanceRecord& l, const DistanceRecord& r) const
    {
        return (l.second < r.second);
    }
};

struct KernelType
{
};

template <class RandomAccessIterator, class Callback> struct KernelDistance
{
    KernelDistance(const Callback& cb) : callback(cb)
    {
    }
    inline ScalarType operator()(const RandomAccessIterator& l, const RandomAccessIterator& r)
    {
        return callback.kernel(*l, *r);
    }
    inline ScalarType distance(const RandomAccessIterator& l, const RandomAccessIterator& r)
    {
        return sqrt(callback.kernel(*l, *l) - 2 * callback.kernel(*l, *r) + callback.kernel(*r, *r));
    }
    typedef KernelType type;
    Callback callback;
};

struct DistanceType
{
};

template <class RandomAccessIterator, class Callback> struct PlainDistance
{
    PlainDistance(const Callback& cb) : callback(cb)
    {
    }
    inline ScalarType operator()(const RandomAccessIterator& l, const RandomAccessIterator& r)
    {
        return callback.distance(*l, *r);
    }
    inline ScalarType distance(const RandomAccessIterator& l, const RandomAccessIterator& r)
    {
        return callback.distance(*l, *r);
    }
    typedef DistanceType type;
    Callback callback;
};

#ifdef TAPKEE_USE_LGPL_COVERTREE
template <class RandomAccessIterator, class Callback>
Neighbors find_neighbors_covertree_impl(RandomAccessIterator begin, RandomAccessIterator end, Callback callback,
                                        IndexType k)
{
    timed_context context("Covertree-based neighbors search");

    typedef CoverTreePoint<RandomAccessIterator> TreePoint;
    v_array<TreePoint> points;
    for (RandomAccessIterator iter = begin; iter != end; ++iter)
        push(points, TreePoint(iter, callback(iter, iter)));

    CoverTreeWrapper<TreePoint, Callback> cover_tree;

    node<TreePoint> ct = cover_tree.batch_create(callback, points);

    v_array<v_array<TreePoint>> res;
    ++k; // because one of the neighbors will be the actual query point
    cover_tree.k_nearest_neighbor(callback, ct, ct, res, k);

    Neighbors neighbors;
    neighbors.resize(end - begin);
    assert(end - begin == res.index);
    for (int i = 0; i < res.index; ++i)
    {
        LocalNeighbors local_neighbors;
        local_neighbors.reserve(k);

        for (IndexType j = 1; j <= k; ++j) // j=0 is the query point
        {
            // The actual query point is found as a neighbor, just ignore it
            if (res[i][j].iter_ - begin == res[i][0].iter_ - begin)
                continue;
            local_neighbors.push_back(res[i][j].iter_ - begin);
        }
        neighbors[res[i][0].iter_ - begin] = local_neighbors;
    };
    return neighbors;
}
#endif

template <class RandomAccessIterator, class Callback>
Neighbors find_neighbors_bruteforce_impl(const RandomAccessIterator& begin, const RandomAccessIterator& end,
                                         Callback callback, IndexType k)
{
    timed_context context("Distance sorting based neighbors search");
    typedef std::pair<RandomAccessIterator, ScalarType> DistanceRecord;
    typedef std::vector<DistanceRecord> Distances;

    Neighbors neighbors;
    neighbors.reserve(end - begin);
    for (RandomAccessIterator iter = begin; iter != end; ++iter)
    {
        Distances distances;
        for (RandomAccessIterator around_iter = begin; around_iter != end; ++around_iter)
            distances.push_back(std::make_pair(around_iter, callback.distance(iter, around_iter)));

        std::nth_element(distances.begin(), distances.begin() + k + 1, distances.end(),
                         distances_comparator<DistanceRecord>());

        LocalNeighbors local_neighbors;
        local_neighbors.reserve(k);
        for (typename Distances::const_iterator neighbors_iter = distances.begin();
             neighbors_iter != distances.begin() + k + 1; ++neighbors_iter)
        {
            if (neighbors_iter->first != iter)
                local_neighbors.push_back(neighbors_iter->first - begin);
        }
        neighbors.push_back(local_neighbors);
    }
    return neighbors;
}

template <class RandomAccessIterator, class Callback>
Neighbors find_neighbors_vptree_impl(const RandomAccessIterator& begin, const RandomAccessIterator& end,
                                     Callback callback, IndexType k)
{
    timed_context context("VP-Tree based neighbors search");

    Neighbors neighbors;
    neighbors.reserve(end - begin);

    VantagePointTree<RandomAccessIterator, Callback> tree(begin, end, callback);

    for (RandomAccessIterator i = begin; i != end; ++i)
    {
        LocalNeighbors local_neighbors = tree.search(i, k + 1);
        auto it = std::remove(local_neighbors.begin(), local_neighbors.end(), i - begin);
        local_neighbors.erase(it, local_neighbors.end());
        neighbors.push_back(local_neighbors);
    }

    return neighbors;
}

template <class RandomAccessIterator, class Callback>
Neighbors find_neighbors(NeighborsMethod method, const RandomAccessIterator& begin, const RandomAccessIterator& end,
                         const Callback& callback, IndexType k, bool check_connectivity)
{
    if (k > static_cast<IndexType>(end - begin - 1))
    {
        Logging::instance().message_warning("Number of neighbors is greater than number of objects to embed. "
                                                     "Using greatest possible number of neighbors.");
        k = static_cast<IndexType>(end - begin - 1);
    }
    Logging::instance().message_info("Using the " + get_neighbors_method_name(method) +
                                              " neighbors computation method.");

    Neighbors neighbors;
    if (method.is(Brute))
        neighbors = find_neighbors_bruteforce_impl(begin, end, callback, k);
    if (method.is(VpTree))
        neighbors = find_neighbors_vptree_impl(begin, end, callback, k);
#ifdef TAPKEE_USE_LGPL_COVERTREE
    if (method.is(CoverTree))
        neighbors = find_neighbors_covertree_impl(begin, end, callback, k);
#endif

    if (check_connectivity && !is_connected(begin, end, neighbors))
    {
        const std::string message = fmt::format("The neighborhood graph with {} neighbors "
                                                "is not connected. Recomputing with a "
                                                "larger number of neighbors {}.",
                                                k, 2 * k);
        Logging::instance().message_warning(message);
        neighbors = find_neighbors(method, begin, end, callback, 2 * k, check_connectivity);
    }
    else
    {
        const std::string message = fmt::format("The neighborhood graph with {} neighbors "
                                                "is connected.",
                                                k);
        Logging::instance().message_info(message);
    }
    return neighbors;
}

} // namespace tapkee_internal
} // namespace tapkee
