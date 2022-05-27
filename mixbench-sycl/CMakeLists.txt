# required cmake version
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(mixbench-sycl LANGUAGES CXX)

# Set default build type to RelWithDebInfo if not specified
if (NOT CMAKE_BUILD_TYPE)
    message (STATUS "Default CMAKE_BUILD_TYPE not set. Using Release with Debug Info")
    set (CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE
        STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel"
        FORCE)
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

add_executable(mixbench-sycl-ro main-sycl.cpp lsyclutil.h mix_kernels_sycl_ro.cpp)
add_executable(mixbench-sycl-alt main-sycl.cpp lsyclutil.h mix_kernels_sycl_alt.cpp)

set_target_properties(mixbench-sycl-ro PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(mixbench-sycl-alt PROPERTIES CXX_EXTENSIONS OFF)

target_compile_definitions(mixbench-sycl-ro PRIVATE READONLY)
