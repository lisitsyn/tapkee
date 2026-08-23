/** Stichwort
 *
 * Copyright (c) 2013, Sergey Lisitsyn <lisitsyn@hey.com>
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include <stichwort/exceptions.hpp>

#include <any>
#include <sstream>
#include <string>

namespace stichwort
{
namespace stichwort_internal
{

class ValueKeeper
{

  public:
    template <typename T> explicit ValueKeeper(const T& v) : value(v), repr_fn(&reprImpl<T>)
    {
    }

    ValueKeeper() : value(), repr_fn(nullptr)
    {
    }

    template <typename T> inline T getValue() const
    {
        if (!isInitialized())
            throw missed_parameter_error("Parameter is missed");

        try
        {
            return std::any_cast<T>(value);
        }
        catch (const std::bad_any_cast&)
        {
            throw wrong_parameter_type_error("Wrong value type");
        }
    }

    template <typename T> inline bool isTypeCorrect() const
    {
        return value.type() == typeid(T);
    }

    inline bool isInitialized() const
    {
        return value.has_value();
    }

    template <template <class> class F, class Q> inline bool isCondition(F<Q> cond) const
    {
        return cond(getValue<Q>());
    }

    inline std::string repr() const
    {
        return isInitialized() ? repr_fn(value) : "uninitialized";
    }

  private:
    template <typename T> static std::string reprImpl(const std::any& v)
    {
        if constexpr (requires(std::stringstream& ss, const T& t) { ss << t; })
        {
            std::stringstream ss;
            ss << std::any_cast<const T&>(v);
            return ss.str();
        }
        else
        {
            return "(can't obtain value)";
        }
    }

    std::any value;
    std::string (*repr_fn)(const std::any&);
};

} // namespace stichwort_internal
} // namespace stichwort
