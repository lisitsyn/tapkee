project (precomputed)

add_executable (precomputed precomputed.cpp)

if (ARPACK_FOUND)
	target_link_libraries(precomputed arpack)
endif()

target_link_libraries(precomputed PRIVATE "${FMT_LIBRARY_TO_LINK}")
