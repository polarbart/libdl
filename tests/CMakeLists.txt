project(libdl_tests)

file(GLOB_RECURSE LIBDL_TEST_FILES *.cpp)
add_executable(${LIBDL_TESTS} ${LIBDL_TEST_FILES})
target_link_libraries(${LIBDL_TESTS} PRIVATE ${LIBDL_SRC} PRIVATE Catch2::Catch2 PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${LIBDL_TESTS} PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_features(${LIBDL_TESTS} PUBLIC cxx_std_17)

include(CTest)
include(Catch)
catch_discover_tests(${LIBDL_TESTS})
