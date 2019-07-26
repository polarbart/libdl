project(libdl_python)

pybind11_add_module(${LIBDL_PYTHON} PythonBindings.cpp)
target_link_libraries(${LIBDL_PYTHON} PRIVATE ${LIBDL_SRC} PRIVATE Eigen3::Eigen PRIVATE pybind11::pybind11)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIBDL_PYTHON} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(${LIBDL_PYTHON} PUBLIC cxx_std_17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation -O3 -march=native")
