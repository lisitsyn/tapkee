#
#
# Locate and configure the Google Mock (and bundled Google Test) libraries for minimal setup
# Tested with google mock version: 1.6, Win32 (VC.NET 2003) & Linux (gcc 4.x.x)
#
# Defines the following variables:
#
#   GMOCKMINIMAL_FOUND - Found the Google Mock libraries
#   GMOCKMINIMAL_INCLUDE_DIRS - The directories needed for include paths
#   GMOCKMINIMAL_SRC - The minimal set off .cc files to use with an executable (mock+test)
#
# Copyright 2011 Amanjit Gill (amanjit.gill@gmx.de)
# Based on a CMake macro from Chandler Carruth
#
#	Licensed under the Apache License, Version 2.0 (the "License"); you may not
#	use this file except in compliance with the License.  You may obtain a copy
#	of the License at
#
#		http://www.apache.org/licenses/LICENSE-2.0
#
#	Unless required by applicable law or agreed to in writing, software
#	distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#	WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
#	License for the specific language governing permissions and limitations
#	under the License.
#
#

if(GMOCKMINIMAL_INCLUDE_DIRS)
  set(GMOCKMINIMAL_FOUND true)

else(GMOCKMINIMAL_INCLUDE_DIRS)
  set(GMOCK_SOURCE_ROOT "" CACHE PATH "Source folder for Google Mock")

  if(GMOCK_SOURCE_ROOT)
    find_path(_GMOCKMINIMAL_INCLUDE_DIR gmock/gmock.h
      PATHS "${GMOCK_SOURCE_ROOT}/include"
      PATH_SUFFIXES ""
      NO_DEFAULT_PATH)

    find_path(_GTESTMINIMAL_INCLUDE_DIR gtest/gtest.h
      PATHS "${GMOCK_SOURCE_ROOT}/gtest/include"
      PATH_SUFFIXES ""
      NO_DEFAULT_PATH)

    find_file(_GMOCKMINIMAL_SRC /src/gmock-all.cc
      PATHS "${GMOCK_SOURCE_ROOT}"
      PATH_SUFFIXES ""
      NO_DEFAULT_PATH)

    find_file(_GTESTMINIMAL_SRC /gtest/src/gtest-all.cc
      PATHS "${GMOCK_SOURCE_ROOT}"
      PATH_SUFFIXES ""
      NO_DEFAULT_PATH)

  else(GMOCK_SOURCE_ROOT)
    find_path(_GMOCKMINIMAL_INCLUDE_DIR gmock/gmock.h
      PATH_SUFFIXES "")
    find_path(_GTESTMINIMAL_INCLUDE_DIR gtest/include/gtest.h
      PATH_SUFFIXES "")
    find_path(_GMOCKMINIMAL_SRC src/gmock-all.cc
      PATH_SUFFIXES "")
    find_path(_GTESTMINIMAL_SRC gtest/src/gtest-all.cc
      PATH_SUFFIXES "")

  endif(GMOCK_SOURCE_ROOT)

  if(_GMOCKMINIMAL_INCLUDE_DIR)
    set(GMOCKMINIMAL_FOUND true)
    set(GMOCKMINIMAL_INCLUDE_DIRS ${_GMOCKMINIMAL_INCLUDE_DIR} ${_GTESTMINIMAL_INCLUDE_DIR} 
      ${GMOCK_SOURCE_ROOT} ${GMOCK_SOURCE_ROOT}/gtest CACHE PATH
      "Include directories for Google Mock library")
    set(GMOCKMINIMAL_SRC ${_GMOCKMINIMAL_SRC} ${_GTESTMINIMAL_SRC} CACHE PATH 
      "Source paths for Google Mock / Google Test combined cpp file (gmock-all.cc and gtest-all.cc)")
    mark_as_advanced(GMOCKMINIMAL_INCLUDE_DIRS)
    mark_as_advanced(GMOCKMINIMAL_SRC)
    if(NOT GoogleMockMinimal_FIND_QUIETLY)
      message(STATUS "Found minimal setup for the Google Mock library: ${GMOCK_SOURCE_ROOT}")
    endif(NOT GoogleMockMinimal_FIND_QUIETLY)

  else(_GMOCKMINIMAL_INCLUDE_DIR)
    if(GoogleMockMinimal_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find the Google Mock library")
    endif(GoogleMockMinimal_FIND_REQUIRED)
  endif(_GMOCKMINIMAL_INCLUDE_DIR)

endif(GMOCKMINIMAL_INCLUDE_DIRS)
