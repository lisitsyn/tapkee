/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_CALLBACK_TRAITS_H_
#define TAPKEE_CALLBACK_TRAITS_H_

template <class Callback>
struct BatchCallbackTraits
{
	static const bool supports_batch;
};
#define TAPKEE_CALLBACK_SUPPORTS_BATCH(X)                            \
template<> const bool BatchCallbackTraits<X>::supports_batch = true; \

#endif
