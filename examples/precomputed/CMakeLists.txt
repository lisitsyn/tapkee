project (precomputed)

add_executable (precomputed precomputed.cpp)

if (ARPACK_FOUND)
	target_link_libraries(precomputed arpack)
endif()


