/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/tapkee_defines.hpp>
/* End of Tapkee includes */

template <class RandomAccessIterator> struct neighbors_finder
{
    virtual Neighbors find_neighbors(RandomAccessIterator begin, RandomAccessIterator end, IndexType k);
};
