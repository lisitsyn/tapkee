project (Tapkee)

cmake_minimum_required (VERSION 2.6.2)

set (TAPKEE_INCLUDE_DIR "tapkee")
set (TAPKEE_SRC_DIR "src")
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

find_package(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")
find_package(Arpack REQUIRED)

include_directories("${TAPKEE_INCLUDE_DIR}")

set (SRCS ${TAPKEE_SRC_DIR}/main.cpp)
add_executable(tapkee ${SRCS})
target_link_libraries(tapkee arpack)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CMAKE_EXE_LINKER_FLAGS "-s")
endif()
