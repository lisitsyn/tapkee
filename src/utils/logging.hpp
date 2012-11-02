/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn
 */

#ifndef EDRT_LOGGING_H_
#define EDRT_LOGGING_H_

#include <iostream>
#include <string>

using std::cout;
using std::ostream;
using std::string;

class LoggingSingleton
{
	private:
		LoggingSingleton() : os_(&cout) {};
		LoggingSingleton(const LoggingSingleton& ls);
		void operator=(const LoggingSingleton& ls);

		ostream* os_;

	public:
		static LoggingSingleton& instance()
		{
			static LoggingSingleton s;
			return s;
		}

		void benchmark(const string& msg) const
		{
			if (os_ && os_->good())
				(*os_) << "[BENCHMARK] " << msg << "\n"; 
		}
};

#endif
