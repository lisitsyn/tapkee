project (Tapkee)

cmake_minimum_required (VERSION 2.6.2)

# set paths
set (TAPKEE_INCLUDE_DIR "tapkee")
set (TAPKEE_SRC_DIR "src")
set (TAPKEE_TESTS_DIR "test")
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

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

# Google Mock detection
find_package(GoogleMockMinimal)
if (GMOCKMINIMAL_FOUND)
	include_directories("${GMOCKMINIMAL_INCLUDE_DIRS}")
endif()

# Main executable
include_directories("${TAPKEE_INCLUDE_DIR}")
set (SRCS ${TAPKEE_SRC_DIR}/main.cpp)
add_executable(tapkee ${SRCS})
if (ARPACK_FOUND)
	target_link_libraries(tapkee arpack)
else()
	add_definitions(-DTAPKEE_NO_ARPACK)
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

# Testing
if (build_tests)
	enable_testing()
	set (TESTS ${TAPKEE_TESTS_DIR}/simple.cpp)
	add_executable(tapkee_tests ${TESTS})
	target_link_libraries(tapkee_tests gmock gmock_main arpack)
	add_test(tapkee tapkee_tests)
endif()
