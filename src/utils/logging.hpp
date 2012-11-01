#ifndef logging_h_
#define logging_h_

#include <iostream>
#include <string>

using std::cout;
using std::ostream;
using std::string;

class LoggingSingleton
{
	private:
		LoggingSingleton()
		{
			os_ = &cout;
		}
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
