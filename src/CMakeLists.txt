project(libdl_src)

file(GLOB_RECURSE LIBDL_SRC_FILES *.cpp *.h)

add_library(${LIBDL_SRC} STATIC ${LIBDL_SRC_FILES})
target_link_libraries(${LIBDL_SRC} PRIVATE Eigen3::Eigen PRIVATE pybind11::pybind11)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIBDL_SRC} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(${LIBDL_SRC} PUBLIC cxx_std_17)