cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mixbench-cpu LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
include_directories("../include")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  MESSAGE(STATUS "GNU C++ compiler is used")
else()
  MESSAGE(WARNING "Not a GNU C++ compiler is used. Potentially not optimized for this.")
endif()

find_package(OpenMP)

option(BASELINE_IMPL "Use baseline implementation" OFF)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
endif()

if (NOT CMAKE_CXX_FLAGS)
    message (STATUS "CMAKE_CXX_FLAGS not set. Using -march=native -funroll-loops")
    set (CMAKE_CXX_FLAGS "-march=native -funroll-loops")
endif()

# Get version info from git tag
execute_process(COMMAND git describe --tags --always
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "")
endif()
string(STRIP "${GIT_REV}" GIT_REV)
file(WRITE "version_info.h" "#define VERSION_INFO \"")
file(APPEND "version_info.h" ${GIT_REV})
file(APPEND "version_info.h" "\"")

add_executable(mixbench-cpu main.cpp mix_kernels_cpu.cpp)

if(${BASELINE_IMPL})
    message(WARNING "Enforcing baseline implementation")
    target_compile_definitions(mixbench-cpu PRIVATE BASELINE_IMPL)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(mixbench-cpu PUBLIC OpenMP::OpenMP_CXX)
endif()
