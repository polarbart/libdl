cmake_minimum_required(VERSION 3.9)

project(libdl VERSION 1.0
        DESCRIPTION "My awsome deep learning library"
        LANGUAGES CXX)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/Catch2" OR
   NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/Eigen" OR
   NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/pybind11")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

enable_testing()
# add_subdirectory(${PROJECT_SOURCE_DIR}/libs/Eigen)
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-header-filter=.;-checks=-*,clang-analyzer-*,clang-analyzer-cplusplus*,modernize-*,performance-*,openmp-*")

# list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/libs/Eigen)
# set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/libs/Eigen/cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/libs/Catch2)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
add_subdirectory(${PROJECT_SOURCE_DIR}/libs/pybind11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/libs/Catch2/contrib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/pylibdl/bin/$<0:>)

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation -O3 -march=native")
endif (UNIX)

if(MSVC)
    add_compile_options(/wd4305)
    add_compile_options(/wd4267)
    add_compile_options(/wd4554)
endif(MSVC)

set(LIBDL_SRC libdl_src)
set(LIBDL_PYTHON libdl_python)
set(LIBDL_TESTS libdl_tests)

add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(tests)
