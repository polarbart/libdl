cmake_minimum_required(VERSION 3.9)

project(libdl VERSION 0.1
        DESCRIPTION "My first DL library"
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
   NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/pybind11" OR
   NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/spdlog")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

set(TEST_TARGET main)

add_executable(${TEST_TARGET} main.cpp src/ComputationalNode.cpp src/ComputationalNode.h src/Add.cpp src/Add.h src/Variable.cpp src/Variable.h src/Multiply.cpp src/Multiply.h src/Sigmoid.cpp src/Sigmoid.h src/CWiseMultiply.cpp src/CWiseMultiply.h src/Sub.cpp src/Sub.h src/ReduceSum.cpp src/ReduceSum.h)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

add_subdirectory(${PROJECT_SOURCE_DIR}/libs/Catch2)
target_link_libraries(${TEST_TARGET} Catch2::Catch2)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(${TEST_TARGET} Eigen3::Eigen)

add_subdirectory(${PROJECT_SOURCE_DIR}/libs/pybind11)
target_link_libraries(${TEST_TARGET} pybind11::pybind11)
# pybind11_add_module(${TEST_TARGET} mymain.cpp)

add_subdirectory(${PROJECT_SOURCE_DIR}/libs/spdlog)
target_link_libraries(${TEST_TARGET} spdlog::spdlog)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/libs/Catch2/contrib")
include(CTest)
include(Catch)
# catch_discover_tests(${TEST_TARGET})

