cmake_minimum_required(VERSION 3.21)

project(mixbench-hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(hip REQUIRED CONFIG)

# Get version info from git tag
execute_process(
    COMMAND git describe --tags --always
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_REV
    ERROR_QUIET
)

if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "")
endif()
string(STRIP "${GIT_REV}" GIT_REV)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/version_info.h"
    "#define VERSION_INFO \"${GIT_REV}\"\n"
)

add_executable(mixbench-hip
    main-hip.cpp
    mix_kernels_hip.cpp
)

set_source_files_properties(
    mix_kernels_hip.cpp
    PROPERTIES LANGUAGE HIP
)

target_include_directories(mixbench-hip PRIVATE
    ../include
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(mixbench-hip PRIVATE hip::host)
