project (minimal)

add_executable (minimal minimal.cpp)

if (ARPACK_FOUND)
	target_link_libraries(minimal arpack)
endif()

target_link_libraries(minimal PRIVATE "${FMT_LIBRARY_TO_LINK}")
