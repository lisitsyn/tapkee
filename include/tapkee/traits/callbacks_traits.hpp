/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

namespace tapkee
{

template <class T> class is_dummy
{
    typedef char yes;
    typedef long no;

    template <typename C> static yes dummy(typename C::dummy*);
    template <typename C> static no dummy(...);

  public:
    static const bool value = (sizeof(dummy<T>(0)) == sizeof(yes));
};

} // namespace tapkee
