/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

#include <queue>
#include <vector>

namespace tapkee
{
namespace tapkee_internal
{

// #pragma GCC diagnostic ignored "-Weffc++"
template <class T, class Comparator>
class reservable_priority_queue : public std::priority_queue<T, std::vector<T>, Comparator>
{
  public:
    typedef typename std::priority_queue<T>::size_type size_type;
    reservable_priority_queue(size_type initial_capacity = 0)
    {
        reserve(initial_capacity);
    }
    void reserve(size_type initial_capacity)
    {
        this->c.reserve(initial_capacity);
    }
    size_type capacity() const
    {
        return this->c.capacity();
    }
    void clear()
    {
        this->c.clear();
    }
};
// #pragma GCC diagnostic pop

} /* End of namespace tapkee_internal */
} /* End of namespace tapkee */
