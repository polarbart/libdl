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



add_subdirectory(${PROJECT_SOURCE_DIR}/libs/Catch2)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_subdirectory(${PROJECT_SOURCE_DIR}/libs/pybind11)
add_subdirectory(${PROJECT_SOURCE_DIR}/libs/spdlog)

set(TEST_TARGET libdl)

pybind11_add_module(${TEST_TARGET} src/Tensor.cpp src/Add.cpp src/Sub.cpp src/MatMul.cpp src/CNode.cpp src/CNodeBase.cpp src/Utils.cpp src/Leaf.cpp src/Sigmoid.cpp src/Pow.cpp src/Sum.cpp)

# add_executable(${TEST_TARGET} main.cpp src/Tensor.cpp src/Add.cpp src/Sub.cpp src/MatMul.cpp src/CNode.cpp src/CNodeBase.cpp src/Utils.cpp src/Leaf.cpp src/Sigmoid.cpp src/Pow.cpp src/Sum.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE Catch2::Catch2 Eigen3::Eigen pybind11::pybind11 spdlog::spdlog)


target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)


# list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/libs/Catch2/contrib")
# include(CTest)
# include(Catch)
# catch_discover_tests(${TEST_TARGET})


