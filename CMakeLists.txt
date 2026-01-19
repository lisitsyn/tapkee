cmake_minimum_required (VERSION 3.24)

project (Tapkee  LANGUAGES CXX)

# set paths
set (CMAKE_CXX_STANDARD 23)
set (TAPKEE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set (TAPKEE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set (TAPKEE_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/unit")
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

# Eigen 5 detection
find_package(Eigen3 5.0 REQUIRED NO_MODULE)
set (EIGEN3_LIBRARY_TO_LINK "Eigen3::Eigen")
message(STATUS "Found Eigen version: ${Eigen3_VERSION}")

# fmt
find_package(fmt REQUIRED)
set (FMT_LIBRARY_TO_LINK "fmt::fmt-header-only")

# cxxopts
find_package(cxxopts REQUIRED)

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
target_include_directories(tapkee_library INTERFACE "${TAPKEE_INCLUDE_DIR}")

if (ARPACK_FOUND)
	target_link_libraries(tapkee PUBLIC arpack)
	target_link_libraries(tapkee_library INTERFACE arpack)
	add_definitions(-DTAPKEE_WITH_ARPACK)
endif()

if (VIENNACL_FOUND)
	target_link_libraries(tapkee PUBLIC OpenCL)
	target_link_libraries(tapkee_library INTERFACE OpenCL)
	add_definitions(-DTAPKEE_WITH_VIENNACL)
endif()

target_link_libraries(tapkee PRIVATE "${FMT_LIBRARY_TO_LINK}")
target_link_libraries(tapkee_library INTERFACE "${FMT_LIBRARY_TO_LINK}")
target_link_libraries(tapkee PRIVATE "${EIGEN3_LIBRARY_TO_LINK}")
target_link_libraries(tapkee_library INTERFACE "${EIGEN3_LIBRARY_TO_LINK}")

if (TAPKEE_CUSTOM_INSTALL_DIR)
	set (TAPKEE_INSTALL_DIR
		"${TAPKEE_CUSTOM_INSTALL_DIR}")
else()
	set (TAPKEE_INSTALL_DIR
		"${CMAKE_INSTALL_PREFIX}/include/tapkee")
endif()

install(
	DIRECTORY "${TAPKEE_INCLUDE_DIR}/tapkee"
	DESTINATION "${TAPKEE_INSTALL_DIR}"
)

install(
	DIRECTORY "${TAPKEE_INCLUDE_DIR}/stichwort"
	DESTINATION "${TAPKEE_INSTALL_DIR}"
)

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

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

# Part of nanobind's setting up the build system, configuring optimized build
# unless otherwise specified, to avoid slow binding code and large binaries.
# https://nanobind.readthedocs.io/en/latest/building.html#preliminaries
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
	add_definitions(-DTAPKEE_DEBUG)
endif()

option(GPL_FREE "Build without GPL-licensed components" OFF)

if (NOT GPL_FREE)
	add_definitions(-DTAPKEE_USE_LGPL_COVERTREE)
endif()

option(BUILD_TESTS "Build tests" OFF)

if (BUILD_TESTS)
	find_package(GTest REQUIRED)
	enable_testing()

	aux_source_directory("${TAPKEE_TESTS_DIR}" "TAPKEE_TESTS_SOURCES")
	foreach(i ${TAPKEE_TESTS_SOURCES})
		get_filename_component(exe ${i} NAME_WE)
		add_executable(test_${exe} ${i})
		target_link_libraries(test_${exe} PUBLIC GTest::gtest GTest::gtest_main)
		target_link_libraries(test_${exe} PRIVATE ${FMT_LIBRARY_TO_LINK})
		target_link_libraries(test_${exe} PRIVATE ${EIGEN3_LIBRARY_TO_LINK})
		if (ARPACK_FOUND)
			target_link_libraries(test_${exe} PUBLIC arpack)
		endif()
		if (VIENNACL_FOUND)
			target_link_libraries(test_${exe} PUBLIC OpenCL)
		endif()
		add_test(
			NAME ${exe}
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${exe}"
			--gtest_color=yes)
	endforeach()
endif()

option(BUILD_NANOBIND "Build nanobind Python extension" OFF)

# https://nanobind.readthedocs.io/en/latest/building.html
# TODO for faster nanobind build, I tried removing the
# add_executable tapkee and that quickly didn't work. It would
# be nice if it could be done without requiring a new option
# BUILD_CLI, or so, maybe if there's a CMake command to
# configure not building tapkee inside the if BUILD_NANOBIND.
if (BUILD_NANOBIND)
	message(STATUS "Detecting and configuring nanobind")
	find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
	# Detect the installed nanobind package and import it into CMake
	execute_process(
		COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
		OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_VARIABLE NB_DIR)
	list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
	find_package(nanobind CONFIG REQUIRED)

	# Build extension (the library that'll be imported from the Python interpreter)
	include_directories("${TAPKEE_SRC_DIR}") # about TODO in nanobind_extension.cpp including src/utils.hpp
	nanobind_add_module(pytapkee src/python/nanobind_extension.cpp) # TODO fix paths.
	target_link_libraries(pytapkee PRIVATE "${EIGEN3_LIBRARY_TO_LINK}")
	target_link_libraries(pytapkee PRIVATE arpack) # TODO ARPACK guard; TODO without ARPACK(?)
	target_link_libraries(pytapkee PRIVATE "${FMT_LIBRARY_TO_LINK}")

	# Rename so that it can be imported as tapkee iso pytapkee.
	# TODO can this go into a separate CMake file?
	add_custom_command(TARGET pytapkee POST_BUILD
		COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/rename_pytapkee.cmake
		COMMENT "Renaming nanobind's extension pytapkee*.so to tapkee*.so")
	# Create the custom script to rename the file
	file(WRITE ${CMAKE_BINARY_DIR}/rename_pytapkee.cmake
		" # Find the file that starts with 'pytapkee' and ends with '.so' in the lib directory
file(GLOB PYLIBRARY_FILE \"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/pytapkee*.so\")

# Ensure exactly one file is found
list(LENGTH PYLIBRARY_FILE FILE_COUNT)
if(NOT FILE_COUNT EQUAL 1)
	message(FATAL_ERROR \"Expected exactly one file starting with 'pytapkee' in lib, but found \${FILE_COUNT}\")
endif()

# Get the first (and only) matched file
list(GET PYLIBRARY_FILE 0 SOURCE_FILE)

# Extract the filename from the full path
get_filename_component(FILENAME \${SOURCE_FILE} NAME)

string(REPLACE \"pytapkee\" \"tapkee\" DEST_FILENAME \${FILENAME})

# Construct the full destination path
set(DEST_FILE \"${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/\${DEST_FILENAME}\")

file(RENAME \${SOURCE_FILE} \${DEST_FILE})

if(EXISTS \${DEST_FILE} AND NOT EXISTS \${SOURCE_FILE})
	message(STATUS \"File renamed from \${SOURCE_FILE} to \${DEST_FILE}\")
else()
	message(FATAL_ERROR \"Error renaming file from \${SOURCE_FILE} to \${DEST_FILE}\")
endif()")
  message(STATUS "Detecting and configuring nanobind - done")
endif()

export(
	TARGETS tapkee_library
	NAMESPACE tapkee::
	FILE ${PROJECT_BINARY_DIR}/tapkee.cmake
)
