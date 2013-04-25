project (rna)

add_executable (rna rna.cpp)

if (ARPACK_FOUND)
	target_link_libraries(rna arpack)
endif()


