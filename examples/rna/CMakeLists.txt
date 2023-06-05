project (rna)

add_executable (rna rna.cpp)

if (ARPACK_FOUND)
    target_link_libraries(rna PRIVATE arpack)
endif()

target_link_libraries(rna PRIVATE "${FMT_LIBRARY_TO_LINK}")
