/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
/* End of Tapkee includes */

namespace tapkee
{
template <class Data> struct dummy_features_callback
{
    typedef int dummy;
    inline tapkee::IndexType dimension() const
    {
        throw tapkee::unsupported_method_error("Dummy feature vector callback is set");
    }
    inline void vector(const Data&, tapkee::DenseVector&) const
    {
        throw tapkee::unsupported_method_error("Dummy feature vector callback is set");
    }
};

template <class Data> struct dummy_kernel_callback
{
    typedef int dummy;
    typedef Data ArgType;
    inline tapkee::ScalarType kernel(const Data&, const Data&) const
    {
        throw tapkee::unsupported_method_error("Dummy kernel callback is set");
        return 0.0;
    }
};

template <class Data> struct dummy_distance_callback
{
    typedef int dummy;
    inline tapkee::ScalarType distance(const Data&, const Data&) const
    {
        throw tapkee::unsupported_method_error("Dummy distance callback is set");
        return 0.0;
    }
};
} // namespace tapkee
