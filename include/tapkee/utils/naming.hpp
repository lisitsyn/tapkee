/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

#include <tapkee/defines/methods.hpp>

#include <string>

namespace tapkee
{

/** Returns the name of the provided method */
inline std::string get_method_name(const DimensionReductionMethod& m)
{
    return m.name();
}

/** Returns the name of the provided neighbors method */
inline std::string get_neighbors_method_name(const NeighborsMethod& m)
{
    return m.name();
}

/** Returns the name of the provided eigen method */
inline std::string get_eigen_method_name(const EigenMethod& m)
{
    return m.name();
}

} // namespace tapkee
