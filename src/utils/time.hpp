#ifndef TIME_H_
#define TIME_H_
#include <ctime>
#include <string>
#include <stdio.h>

struct timed_context
{
	clock_t start_clock;
	std::string operation_name;
	timed_context(const std::string& name)
	{
		operation_name = name;
		start_clock = clock();
	}
	~timed_context()
	{
		printf("%s took %f seconds\n",operation_name.c_str(),double(clock()-start_clock)/CLOCKS_PER_SEC);
	}
};
#endif
