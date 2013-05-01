/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_POLICY_H_
#define TAPKEE_POLICY_H_

namespace tapkee
{
namespace tapkee_internal
{

struct TypePolicyBase
{
	virtual ~TypePolicyBase() {}
	virtual void copyFromValue(void const*, void**) const = 0;
	virtual void* getValue(void**) const = 0;
	virtual void free(void**) const = 0;
	virtual void clone(void* const*, void**) const = 0;
	virtual void move(void* const*, void**) const = 0;
};

template <typename T>
struct PointerTypePolicyImpl : public TypePolicyBase
{
	inline virtual void copyFromValue(void const* src, void** dest) const
	{
		*dest = new T(*reinterpret_cast<T const*>(src));
	}
	inline virtual void* getValue(void** src) const
	{
		return *src;
	}
	inline virtual void free(void** src) const
	{
		if (*src) 
			delete (*reinterpret_cast<T**>(src));
		*src = NULL;
	}
	virtual void clone(void* const* src, void** dest) const
	{
		if (*dest) 
			(*reinterpret_cast<T**>(dest))->~T();
		*dest = new T(**reinterpret_cast<T* const*>(src));
	}
	inline virtual void move(void* const* src, void** dest) const
	{
		(*reinterpret_cast<T**>(dest))->~T();
		**reinterpret_cast<T**>(dest) = **reinterpret_cast<T* const*>(src); 
	}
};

template <typename T>
TypePolicyBase* getPolicy()
{
	static PointerTypePolicyImpl<T> policy;
	return &policy;
}

struct CheckerPolicyBase
{
	virtual ~CheckerPolicyBase() {}
	virtual bool isInRange(void* const*, void*, void*) const = 0;
	virtual bool isEqual(void* const*, void*) const = 0;
	virtual bool isNotEqual(void* const*, void*) const = 0;
	virtual bool isPositive(void* const*) const = 0;
	virtual bool isNonNegative(void * const*) const = 0;
	virtual bool isNegative(void* const*) const = 0;
	virtual bool isNonPositive(void * const*) const = 0;
	virtual bool isGreater(void* const*, void*) const = 0;
	virtual bool isLesser(void* const*, void*) const = 0;
};

template <typename T>
struct PointerCheckerPolicyImpl : public CheckerPolicyBase
{
	inline T value(void* v) const
	{
		return *reinterpret_cast<T*>(v);
	}
	virtual bool isInRange(void* const* src, void* lower, void* upper) const
	{
		T v = value(*src);
		T l = value(lower);
		T u = value(upper);
		return (v>=l) && (v<u);
	}
	virtual bool isEqual(void* const* src, void* other_src) const
	{
		T v = value(*src);
		T ov = value(other_src);
		return (v==ov);
	}
	virtual bool isNotEqual(void* const* src, void* other_src) const
	{
		T v = value(*src);
		T ov = value(other_src);
		return (v!=ov);
	}
	virtual bool isPositive(void* const* src) const
	{
		T v = value(*src);
		return (v>0);
	}
	virtual bool isNonNegative(void* const* src) const
	{
		T v = value(*src);
		return (v>=0);
	}
	virtual bool isNegative(void* const* src) const
	{
		T v = value(*src);
		return (v<0);
	}
	virtual bool isNonPositive(void* const* src) const
	{
		T v = value(*src);
		return (v<=0);
	}
	virtual bool isGreater(void* const* src, void* lower) const
	{
		T v = value(*src);
		return (v>value(lower));
	}
	virtual bool isLesser(void* const* src, void* upper) const
	{
		T v = value(*src);
		return (v<value(upper));
	}
};

struct EmptyType;

template <>
struct PointerCheckerPolicyImpl<EmptyType> : public CheckerPolicyBase
{
	virtual bool isInRange(void* const*, void*, void*) const
	{
		return false;
	}
	virtual bool isEqual(void* const*, void*) const
	{
		return false;
	}
	virtual bool isNotEqual(void* const*, void*) const
	{
		return false;
	}
	virtual bool isPositive(void* const*) const
	{
		return false;
	}
	virtual bool isNonNegative(void* const*) const
	{
		return false;
	}
	virtual bool isNegative(void* const*) const
	{
		return false;
	}
	virtual bool isNonPositive(void* const*) const
	{
		return false;
	}
	virtual bool isGreater(void* const*, void*) const
	{
		return false;
	}
	virtual bool isLesser(void* const*, void*) const
	{
		return false;
	}
};

template <>
struct PointerCheckerPolicyImpl<bool> : public CheckerPolicyBase
{
	inline bool value(void* v) const
	{
		return *reinterpret_cast<bool*>(v);
	}
	virtual bool isInRange(void* const*, void*, void*) const
	{
		return false;
	}
	virtual bool isEqual(void* const* src, void* other_src) const
	{
		bool v = value(*src);
		bool ov = value(other_src);
		return (v==ov);
	}
	virtual bool isNotEqual(void* const* src, void* other_src) const
	{
		bool v = value(*src);
		bool ov = value(other_src);
		return (v!=ov);
	}
	virtual bool isPositive(void* const*) const
	{
		return false;
	}
	virtual bool isNonNegative(void* const*) const
	{
		return false;
	}
	virtual bool isNegative(void* const*) const
	{
		return false;
	}
	virtual bool isNonPositive(void* const*) const
	{
		return false;
	}
	virtual bool isGreater(void* const*, void*) const
	{
		return false;
	}
	virtual bool isLesser(void* const*, void*) const
	{
		return false;
	}
};

template <typename T>
CheckerPolicyBase* getCheckerPolicy()
{
	static PointerCheckerPolicyImpl<T> policy;
	return &policy;
}

}
}
#endif
