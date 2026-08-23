/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

#include <algorithm>
#include <limits>
#include <random>

namespace tapkee
{

inline std::mt19937_64& random_generator()
{
    static thread_local std::mt19937_64 generator(std::random_device{}());
    return generator;
}

inline IndexType uniform_random_index()
{
#ifdef CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION
    return CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION % std::numeric_limits<IndexType>::max();
#else
    std::uniform_int_distribution<IndexType> distribution(0, std::numeric_limits<IndexType>::max() - 1);
    return distribution(random_generator());
#endif
}

inline IndexType uniform_random_index_bounded(IndexType upper)
{
#ifdef CUSTOM_UNIFORM_RANDOM_INDEX_FUNCTION
    return uniform_random_index() % upper;
#else
    std::uniform_int_distribution<IndexType> distribution(0, upper - 1);
    return distribution(random_generator());
#endif
}

inline ScalarType uniform_random()
{
#ifdef CUSTOM_UNIFORM_RANDOM_FUNCTION
    return CUSTOM_UNIFORM_RANDOM_FUNCTION;
#else
    std::uniform_real_distribution<ScalarType> distribution(0.0, 1.0);
    return distribution(random_generator());
#endif
}

inline ScalarType gaussian_random()
{
#ifdef CUSTOM_GAUSSIAN_RANDOM_FUNCTION
    return CUSTOM_GAUSSIAN_RANDOM_FUNCTION;
#else
    std::normal_distribution<ScalarType> distribution(0.0, 1.0);
    return distribution(random_generator());
#endif
}

template <class RAI> inline void random_shuffle(RAI first, RAI last)
{
    std::shuffle(first, last, random_generator());
}

} // namespace tapkee
