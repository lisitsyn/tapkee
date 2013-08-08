/** Stichwort
 *
 * Copyright (c) 2013, Sergey Lisitsyn <lisitsyn.s.o@gmail.com>
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

#ifndef STICHWORT_KEEPER_H_
#define STICHWORD_KEEPER_H_

#include <stichwort/policy.hpp>
#include <stichwort/exceptions.hpp>

namespace stichwort
{
namespace stichwort_internal
{

struct EmptyType
{
};

class ValueKeeper
{

public:
	template <typename T>
	explicit ValueKeeper(const T& value) : 
		policy(getPolicy<T>()), checker(getCheckerPolicy<T>()), value_ptr(NULL) 
	{
		policy->copyFromValue(&value, &value_ptr);
	}

	ValueKeeper() :
		policy(getPolicy<EmptyType>()), checker(getCheckerPolicy<EmptyType>()), value_ptr(NULL) 
	{
	}

	~ValueKeeper()
	{
		policy->free(&value_ptr);
	}

	ValueKeeper(const ValueKeeper& v) : policy(v.policy), checker(v.checker), value_ptr(NULL)
	{
		policy->clone(&(v.value_ptr), &value_ptr);
	}

	ValueKeeper& operator=(const ValueKeeper& v)
	{
		policy->free(&value_ptr);
		policy = v.policy;
		checker = v.checker;
		policy->clone(&(v.value_ptr), &value_ptr);
		return *this;
	}

	template <typename T>
	inline T getValue() const
	{
		T* v;
		if (!isInitialized())
		{
			throw missed_parameter_error("Parameter is missed");
		}
		if (isTypeCorrect<T>())
		{
			void* vv = policy->getValue(const_cast<void**>(&value_ptr));
			v = reinterpret_cast<T*>(vv);
		}
		else
			throw wrong_parameter_type_error("Wrong value type");
		return *v;
	}

	template <typename T>
	inline bool isTypeCorrect() const
	{
		return getPolicy<T>() == policy;
	}

	inline bool isInitialized() const
	{
		return getPolicy<EmptyType>() != policy;
	}

	template <typename T>
	inline bool inRange(T lower, T upper) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong range bounds type");
		return checker->isInRange(&value_ptr,&lower,&upper);
	}

	template <typename T>
	inline bool equal(T value) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong equality value type");
		return checker->isEqual(&value_ptr,&value);
	}

	template <typename T>
	inline bool notEqual(T value) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong non-equality value type");
		return checker->isNotEqual(&value_ptr,&value);
	}

	inline bool positive() const
	{
		return checker->isPositive(&value_ptr);
	}

	inline bool nonNegative() const
	{
		return checker->isNonNegative(&value_ptr);
	}

	inline bool negative() const
	{
		return checker->isNegative(&value_ptr);
	}

	inline bool nonPositive() const
	{
		return checker->isNonPositive(&value_ptr);
	}

	template <typename T>
	inline bool greater(T lower) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong greater check bound type");
		return checker->isGreater(&value_ptr,&lower);
	}

	template <typename T>
	inline bool lesser(T upper) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong lesser check bound type");
		return checker->isLesser(&value_ptr,&upper);
	}

private:

	TypePolicyBase* policy;
	CheckerPolicyBase* checker;
	void* value_ptr;

};

}
}
#endif
