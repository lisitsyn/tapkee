#ifndef logging_h_
#define logging_h_

#include <iostream>

class LoggingSingleton
{
	private:
		LoggingSingleton()
		{
			
		}
		LoggingSingleton(const LoggingSingleton& ls);
		operator=(const LoggingSingleton& ls);

	public:
		LoggingSingleton& instance()
		{
			static LoggingSingleton s;
			return s;
		}
}

#endif
