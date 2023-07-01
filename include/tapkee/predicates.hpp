/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2023 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/defines.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

template <typename T> struct Positivity
{
    inline bool operator()(T v) const
    {
        return v > 0;
    }
    inline std::string failureMessage(const stichwort::Parameter& p) const
    {
        return fmt::format("Positivity check failed for {}, its value is {}", p.name(), p.repr());
    }
};

template <typename T> struct NonNegativity
{
    inline bool operator()(T v) const
    {
        return v >= 0;
    }
    inline std::string failureMessage(const stichwort::Parameter& p) const
    {
        return fmt::format("Non-negativity check failed for {}, its value is {}", p.name(), p.repr());
    }
};

template <typename T> struct InRange
{
    InRange(T l, T u) : lower(l), upper(u)
    {
    }
    inline bool operator()(T v) const
    {
        return (v >= lower) && (v < upper);
    }
    T lower;
    T upper;
    inline std::string failureMessage(const stichwort::Parameter& p) const
    {
        return fmt::format("[{}, {}) range check failed for {}, its value is {}", lower, upper, p.name(), p.repr());
    }
};

template <typename T> struct InClosedRange
{
    InClosedRange(T l, T u) : lower(l), upper(u)
    {
    }
    inline bool operator()(T v) const
    {
        return (v >= lower) && (v <= upper);
    }
    T lower;
    T upper;
    inline std::string failureMessage(const stichwort::Parameter& p) const
    {
        return fmt::format("[{}, {}] range check failed for {}, its value is {}", lower, upper, p.name(), p.repr());
    }
};

}
}
