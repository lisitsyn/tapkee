/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef TAPKEE_LOGGING_H_
#define TAPKEE_LOGGING_H_

#include <iostream>
#include <string>

using std::cout;
using std::ostream;
using std::string;

#define LEVEL_ENABLED_FIELD(X) bool X##_enabled
#define LEVEL_ENABLED_FIELD_INITIALIZER(X,value) X##_enabled(value)
#define LEVEL_HANDLERS(LEVEL) \
		void enable_##LEVEL() { LEVEL##_enabled = true; };		\
		void disable_##LEVEL() { LEVEL##_enabled = false; };	\
		void message_##LEVEL(const string& msg)					\
		{														\
			if (LEVEL##_enabled && os_ && os_->good())			\
				(*os_) << "["#LEVEL"] " << msg << "\n";			\
		}

class LoggingSingleton
{
	private:
		LoggingSingleton() : os_(&cout),
			LEVEL_ENABLED_FIELD_INITIALIZER(info,false),
			LEVEL_ENABLED_FIELD_INITIALIZER(warning,true),
			LEVEL_ENABLED_FIELD_INITIALIZER(error,true),
			LEVEL_ENABLED_FIELD_INITIALIZER(benchmark,false)
		{
		};
		LoggingSingleton(const LoggingSingleton& ls);
		void operator=(const LoggingSingleton& ls);

		ostream* os_;

		LEVEL_ENABLED_FIELD(info);
		LEVEL_ENABLED_FIELD(warning);
		LEVEL_ENABLED_FIELD(error);
		LEVEL_ENABLED_FIELD(benchmark);

	public:
		static LoggingSingleton& instance()
		{
			static LoggingSingleton s;
			return s;
		}

		LEVEL_HANDLERS(info);
		LEVEL_HANDLERS(warning);
		LEVEL_HANDLERS(error);
		LEVEL_HANDLERS(benchmark);

};

#undef LEVEL_HANDLERS
#undef LEVEL_ENABLED_FIELD

#endif
