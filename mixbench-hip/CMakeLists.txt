cmake_minimum_required(VERSION 2.8.3)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(mixbench-hip)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
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

set(ALT_SOURCE_FILES main-hip.cpp mix_kernels_hip.cpp)
set(RO_SOURCE_FILES main-hip-ro.cpp mix_kernels_hip_ro.cpp)

set_source_files_properties(${ALT_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
set_source_files_properties(${RO_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_executable(mixbench-hip-alt ${ALT_SOURCE_FILES})
hip_add_executable(mixbench-hip-ro ${RO_SOURCE_FILES})

