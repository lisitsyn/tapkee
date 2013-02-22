project (Tapkee)

cmake_minimum_required (VERSION 2.6.2)

# set paths
set (TAPKEE_INCLUDE_DIR "tapkee")
set (TAPKEE_SRC_DIR "src")
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

# Eigen3 detection
find_package(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")

# ARPACK detection
find_package(Arpack REQUIRED)
link_directories("${ARPACK_PATH}")

include_directories("${TAPKEE_INCLUDE_DIR}")
set (SRCS ${TAPKEE_SRC_DIR}/main.cpp)
add_executable(tapkee ${SRCS})
target_link_libraries(tapkee arpack)

add_definitions(-DUSE_PRECOMPUTED)

if (TAPKEE_CUSTOM_INSTALL_DIR)
	set (TAPKEE_INSTALL_DIR
		"${TAPKEE_CUSTOM_INSTALL_DIR}")
else()
	set (TAPKEE_INSTALL_DIR
		"${CMAKE_INSTALL_PREFIX}/include")
endif()

install(DIRECTORY "${TAPKEE_INCLUDE_DIR}" DESTINATION "${TAPKEE_INSTALL_DIR}")

exec_program("git" ARGS "describe --always" OUTPUT_VARIABLE VERSION)
add_definitions(-DGIT_INFO="${VERSION}")

set(CMAKE_BUILD_TYPE Release)
