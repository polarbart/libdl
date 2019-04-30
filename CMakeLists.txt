cmake_minimum_required(VERSION 3.9)
project(libdl VERSION 0.1
        DESCRIPTION "My first DL library"
        LANGUAGES CXX)

set(TEST_TARGET testmain)

add_executable(${TEST_TARGET} main.cpp)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_17)

