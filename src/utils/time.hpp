#ifndef TIME_H_
#define TIME_H_
#include <ctime>
#include <string>
#include <iostream>
#include "logging.hpp"

using std::string;
using std::stringstream;

struct timed_context
{
	clock_t start_clock;
	string operation_name;
	timed_context(const std::string& name)
	{
		operation_name = name;
		start_clock = clock();
	}
	~timed_context()
	{
		stringstream msg_stream;
		msg_stream << operation_name << " took " << double(clock()-start_clock)/CLOCKS_PER_SEC << " seconds"; 
		LoggingSingleton::instance().benchmark(msg_stream.str());
	}
};
#endif
