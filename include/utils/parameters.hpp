/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_PARAMETERS_H_
#define TAPKEE_PARAMETERS_H_

#include <utils/value_keeper.hpp>

namespace tapkee
{
namespace tapkee_internal
{

struct Message
{
	Message() : ss()
	{
	}
	template <typename T>
	Message& operator<<(const T& data)
	{
		ss << data;
		return *this;
	}
	operator std::string() 
	{
		return ss.str();
	}
	std::stringstream ss;
};

class CheckedParameter;

class Parameter
{

	friend class CheckedParameter;

private:

	template <typename T>
	Parameter(const T& value) : keeper(ValueKeeper(value))
	{
	}

public:

	template <typename T>
	static Parameter of(const T& value) 
	{
		return Parameter(value);
	}

	Parameter() : keeper(ValueKeeper())
	{
	}

	Parameter(const Parameter& p) : keeper(p.keeper)
	{
	}

	~Parameter()
	{
	}

	template <typename T>
	inline Parameter withDefault(T value)
	{
		if (!is_initialized())
		{
			keeper = ValueKeeper(value);
		}
		return *this;
	}

	template <typename T>
	inline operator T()
	{
		return value<T>();
	}

	template <typename T>
	bool is(T v)
	{
		if (!is_type_correct<T>())
			return false;
		T kv = keeper.value<T>();
		if (v == kv)
			return true;
		return false;
	}

	template <typename T>
	bool operator==(T v) const
	{
		return is<T>(v);
	}

	CheckedParameter checked();

	template <typename T>
	bool in_range(T lower, T upper) const
	{
		return keeper.in_range<T>(lower, upper);
	}

	template <typename T>
	bool equal(T value) const
	{
		return keeper.equal<T>(value);
	}

	template <typename T>
	bool not_equal(T value) const
	{
		return keeper.not_equal<T>(value);
	}

	bool positive() const 
	{
		return keeper.positive();
	}

	bool negative() const
	{
		return keeper.negative();
	}

	template <typename T>
	bool greater(T lower) const
	{
		return keeper.greater<T>(lower);
	}

	template <typename T>
	bool is_lesser(T upper) const
	{
		return keeper.lesser<T>(upper);
	}

	bool is_initialized() const
	{
		return keeper.is_initialized();
	}

private:

	template <typename T>
	inline T value() const
	{
		return keeper.value<T>();
	}
	
	template <typename T>
	inline bool is_type_correct() const
	{
		return keeper.is_type_correct<T>();
	}

private:

	ValueKeeper keeper; 

};

class CheckedParameter
{

public:

	explicit CheckedParameter(const Parameter& p) : parameter(p)
	{
	}

	template <typename T>
	inline operator T()
	{
		return parameter.value<T>();
	}
	
	inline operator const Parameter&()
	{
		return parameter;
	}

	template <typename T>
	bool is(T v)
	{
		return parameter.is<T>(v);
	}

	template <typename T>
	bool operator==(T v)
	{
		return is<T>(v);
	}

	template <typename T>
	CheckedParameter& in_range(T lower, T upper)
	{
		if (!parameter.in_range(lower, upper))
		{
			std::string error_message = 
				(Message() << "Value " << parameter.value<T>() << " doesn't fit the range [" << 
				 lower << ", " << upper << ")");
			throw tapkee::wrong_parameter_error(error_message);
		}
		return *this;
	}
	
	CheckedParameter& positive()
	{
		if (!parameter.positive())
		{
			std::string error_message = 
				(Message() << "Value is not positive");
			throw tapkee::wrong_parameter_error(error_message);
		}
		return *this;
	}

private:

	Parameter parameter;

};

CheckedParameter Parameter::checked() 
{
	return CheckedParameter(*this);
}

}
}
#endif
