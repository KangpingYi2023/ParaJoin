cmake_minimum_required(VERSION 3.23.1)
project(VECTORJOIN)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set( CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-inline -fno-omit-frame-pointer -fno-eliminate-unused-debug-types")
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -lrt -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")

if( CMAKE_BUILD_TYPE STREQUAL "Release")
    add_definitions(-DRELEASE_BUILD)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(tests)