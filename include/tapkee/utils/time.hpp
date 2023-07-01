/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */
#pragma once

/* Tapkee includes */
#include <tapkee/utils/logging.hpp>
/* End of Tapkee includes */

#include <ctime>
#include <sstream>
#include <string>

namespace tapkee
{
namespace tapkee_internal
{
#ifdef _OPENMP
#define CLOCK_TYPE double
#define CLOCK_GET omp_get_wtime()
#define CLOCK_DIVISOR 1
#else
#define CLOCK_TYPE clock_t
#define CLOCK_GET clock()
#define CLOCK_DIVISOR CLOCKS_PER_SEC
#endif

struct timed_context
{
    CLOCK_TYPE start_clock;
    std::string operation_name;
    timed_context(const std::string& name) : start_clock(CLOCK_GET), operation_name(name)
    {
    }
    ~timed_context()
    {
        std::string message =
            fmt::format("{} took {} seconds.", operation_name, double(CLOCK_GET - start_clock) / CLOCK_DIVISOR);
        LoggingSingleton::instance().message_benchmark(message);
    }
};
} // namespace tapkee_internal
} // namespace tapkee

#undef CLOCK_TYPE
#undef CLOCK_GET
#undef CLOCK_DIVISOR
