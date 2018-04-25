project (Tapkee  LANGUAGES CXX)

cmake_minimum_required (VERSION 3.3)

# set paths
set (CMAKE_CXX_STANDARD 11)
set (TAPKEE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set (TAPKEE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set (TAPKEE_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/unit")
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# Eigen3 detection
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM "${EIGEN3_INCLUDE_DIR}")

# ARPACK detection
find_package(Arpack)
if (ARPACK_FOUND)
	link_directories("${ARPACK_PATH}")
endif()

# OpenMP detection
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ViennaCL detection
find_package(OpenCL)
if (OPENCL_FOUND)
	find_package(ViennaCL)
	if (VIENNACL_FOUND)
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPENCL_C_FLAGS}")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENCL_CXX_FLAGS}")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENCL_EXE_LINKER_FLAGS}")
		include_directories("${VIENNACL_INCLUDE_DIRS}")
	endif()
endif()

include_directories("${TAPKEE_INCLUDE_DIR}")
# CLI executable
add_executable(tapkee "${TAPKEE_SRC_DIR}/cli/main.cpp")
# Examples
option(BUILD_EXAMPLES "Whether to build examples or not" OFF)
if (BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# library interface
file(GLOB_RECURSE headers_library "${TAPKEE_INCLUDE_DIR}/*.hpp")
add_library(tapkee_library INTERFACE)
target_include_directories(tapkee_library INTERFACE "${TAPKEE_INCLUDE_DIR}" "${EIGEN3_INCLUDE_DIR}")

if (ARPACK_FOUND)
	target_link_libraries(tapkee arpack)
	target_link_libraries(tapkee_library INTERFACE arpack)
	add_definitions(-DTAPKEE_WITH_ARPACK)
endif()

if (VIENNACL_FOUND)
	target_link_libraries(tapkee OpenCL)
	target_link_libraries(tapkee_library INTERFACE OpenCL)
	add_definitions(-DTAPKEE_WITH_VIENNACL)
endif()

if (TAPKEE_CUSTOM_INSTALL_DIR)
	set (TAPKEE_INSTALL_DIR
		"${TAPKEE_CUSTOM_INSTALL_DIR}")
else()
	set (TAPKEE_INSTALL_DIR
		"${CMAKE_INSTALL_PREFIX}/include/tapkee")
endif()


file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/utils/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/utils")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/routines/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/routines")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/callbacks/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/callbacks")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/neighbors/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/neighbors")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/traits/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/traits")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/parameters/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/parameters")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/defines/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/defines")
file(GLOB headers "${TAPKEE_INCLUDE_DIR}/tapkee/external/barnes_hut_sne/*.hpp")
install(FILES ${headers} DESTINATION "${TAPKEE_INSTALL_DIR}/external/barnes_hut_sne/")

# G++ specific flags
option(USE_GCOV "Use gcov to analyze test coverage" OFF)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-long-long -Wshadow")
	if (USE_GCOV)
		set(CMAKE_BUILD_TYPE Debug)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
		set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")

		add_custom_target(lcov DEPENDS tapkee)
		add_custom_command(TARGET lcov COMMAND mkdir -p coverage WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
		add_custom_command(TARGET lcov COMMAND lcov --directory . --zerocounters WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
		add_custom_command(TARGET lcov COMMAND make test WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
		add_custom_command(TARGET lcov COMMAND lcov --directory . --capture --output-file ./coverage/out.info WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
		add_custom_command(TARGET lcov COMMAND lcov --remove ./coverage/out.info /usr/local/include\\* /usr/include\\* unit\\* --output ./coverage/clean.info "${CMAKE_BINARY_DIR}" VERBATIM)
		add_custom_command(TARGET lcov COMMAND genhtml -o ./coverage ./coverage/clean.info --branch-coverage --demangle-cpp WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" VERBATIM)
	endif()
endif()

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-DTAPKEE_DEBUG)
endif()

option(PRECOMPUTED "Precompute kernel/distance matrices before reducing dimensionality" OFF)

if (PRECOMPUTED)
	add_definitions(-DUSE_PRECOMPUTED)
endif()

option(GPL_FREE "Build without GPL-licensed components" OFF)

if (NOT GPL_FREE)
	add_definitions(-DTAPKEE_USE_LGPL_COVERTREE)
endif()

option(BUILD_TESTS "Build tests" OFF)

if (BUILD_TESTS)
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest-release-1.8.0/")
		add_subdirectory(googletest-release-1.8.0)
		set(GTEST_SOURCE_DIR "${gtest_SOURCE_DIR}")
		include_directories(SYSTEM "${GTEST_SOURCE_DIR}/include")
	else()
		if (NOT GTEST_SOURCE_DIR)
			# try with default ubuntu location
			set(GTEST_SOURCE_DIR /usr/src/gtest)
		endif()
		if (NOT GTEST_INCLUDE_DIR)
			# try with default ubuntu location
			set(GTEST_INCLUDE_DIR /usr/include/gtest)
		endif()
		if (GTEST_SOURCE_DIR AND GTEST_INCLUDE_DIR)
			include_directories("${GTEST_SOURCE_DIR}" "${GTEST_INCLUDE_DIR}/include")
			add_library(gtest STATIC "${GTEST_SOURCE_DIR}/src/gtest-all.cc")
			add_library(gtest_main STATIC "${GTEST_SOURCE_DIR}/src/gtest_main.cc")
			target_link_libraries(gtest_main gtest)
			find_package(Threads)
			target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})
		else()
			message(FATAL_ERROR "gtest-1.8.0 is required to be available in the root of the project or reachable
			by GTEST_SOURCE_DIR and GTEST_INCLUDE_DIR")
		endif()
	endif()
	enable_testing()

	aux_source_directory("${TAPKEE_TESTS_DIR}" "TAPKEE_TESTS_SOURCES")
	foreach(i ${TAPKEE_TESTS_SOURCES})
		get_filename_component(exe ${i} NAME_WE)
		add_executable(test_${exe} ${i})
		target_link_libraries(test_${exe} gtest gtest_main)
		if (ARPACK_FOUND)
			target_link_libraries(test_${exe} arpack)
		endif()
		if (VIENNACL_FOUND)
			target_link_libraries(test_${exe} OpenCL)
		endif()
		add_test(
			NAME ${exe}
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${exe}"
			--gtest_color=yes)
	endforeach()
endif()
