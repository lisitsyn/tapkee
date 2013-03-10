project (Tapkee)

cmake_minimum_required (VERSION 2.6.2)

# set paths
set (TAPKEE_INCLUDE_DIR "tapkee")
set (TAPKEE_SRC_DIR "src")
set (TAPKEE_TESTS_DIR "test/gtest")
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")
set (EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Eigen3 detection
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM "${EIGEN3_INCLUDE_DIR}")

# ARPACK detection
find_package(Arpack REQUIRED)
link_directories("${ARPACK_PATH}")

# OpenMP detection
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Main executable
include_directories("${TAPKEE_INCLUDE_DIR}")
set (SRCS ${TAPKEE_SRC_DIR}/main.cpp)
add_executable(tapkee ${SRCS})
if (ARPACK_FOUND)
	target_link_libraries(tapkee arpack)
	add_definitions(-DTAPKEE_WITH_ARPACK)
endif()

if (TAPKEE_CUSTOM_INSTALL_DIR)
	set (TAPKEE_INSTALL_DIR
		"${TAPKEE_CUSTOM_INSTALL_DIR}")
else()
	set (TAPKEE_INSTALL_DIR
		"${CMAKE_INSTALL_PREFIX}/include")
endif()

install(DIRECTORY "${TAPKEE_INCLUDE_DIR}" DESTINATION "${TAPKEE_INSTALL_DIR}")

# G++ specific flags
if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++ -Wall -Wextra")
endif()

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

if (PRECOMPUTED)
	add_definitions(-DUSE_PRECOMPUTED)
endif()

if (NOT GPL_FREE)
	#add_definitions(-DTAPKEE_USE_GPL_TSNE)
	add_definitions(-DTAPKEE_USE_LGPL_COVERTREE)
endif()

# Testing
if (BUILD_TESTS)
	if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/gtest-1.6.0/)
		add_subdirectory(gtest-1.6.0)
	else()
		message("gtest-1.6.0 is required to be available in root of the project")
	endif()
	include_directories(SYSTEM ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
	enable_testing()

	aux_source_directory(${TAPKEE_TESTS_DIR} TAPKEE_TESTS_SOURCES)
	foreach(i ${TAPKEE_TESTS_SOURCES})
		get_filename_component(exe ${i} NAME_WE)
		add_executable(test_${exe} ${i})
		target_link_libraries(test_${exe} gtest gtest_main arpack)
		add_test(
			NAME ${exe}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			COMMAND ${EXECUTABLE_OUTPUT_PATH}/test_${exe}
			--gtest_color=yes)
	endforeach()
endif()
