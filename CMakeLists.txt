project (Tapkee)

cmake_minimum_required (VERSION 2.6.2)

# set paths
set (TAPKEE_INCLUDE_DIR "tapkee")
set (TAPKEE_SRC_DIR "src")
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

include_directories("${TAPKEE_INCLUDE_DIR}")
set (SRCS ${TAPKEE_SRC_DIR}/main.cpp)
add_executable(tapkee ${SRCS})
if (ARPACK_FOUND)
	target_link_libraries(arpack)
else()
	add_definitions(-DTAPKEE_NO_ARPACK)
endif()
target_link_libraries(tapkee)

#add_definitions(-DUSE_PRECOMPUTED)

if (TAPKEE_CUSTOM_INSTALL_DIR)
	set (TAPKEE_INSTALL_DIR
		"${TAPKEE_CUSTOM_INSTALL_DIR}")
else()
	set (TAPKEE_INSTALL_DIR
		"${CMAKE_INSTALL_PREFIX}/include")
endif()

install(DIRECTORY "${TAPKEE_INCLUDE_DIR}" DESTINATION "${TAPKEE_INSTALL_DIR}")

#exec_program("git" ARGS "describe --always" OUTPUT_VARIABLE VERSION)
#add_definitions(-DGIT_INFO="${VERSION}")

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++ -Wall -Wextra")
endif()

set(CMAKE_BUILD_TYPE Release)
