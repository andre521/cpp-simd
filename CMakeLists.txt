project(cpp-simd)
cmake_minimum_required(VERSION 2.8)
get_directory_property(HAS_PARENT PARENT_DIRECTORY)

set(PROJECT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CPP_SIMD_INCLUDE "${PROJECT_SOURCE_DIR}")
include_directories(${CPP_SIMD_INCLUDE})
if (HAS_PARENT)
    set(CPP_SIMD_INCLUDE "${CPP_SIMD_INCLUDE}" PARENT_SCOPE)
endif()

if (NOT HAS_PARENT)
    execute_process(COMMAND git submodule update --init test/gtest
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(test/gtest)
    include_directories(${gtest_SOURCE_DIR}/include)

    add_executable(simd_test test/simd)
    target_link_libraries(simd_test gtest)

    enable_testing()
    add_test(simd_test simd_test)
endif()
