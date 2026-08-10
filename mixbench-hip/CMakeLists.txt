cmake_minimum_required(VERSION 3.21)

project(mixbench-hip LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(hip REQUIRED CONFIG)

add_executable(mixbench-hip
    main-hip.cpp
    mix_kernels_hip.cpp
)

set_source_files_properties(
    mix_kernels_hip.cpp
    PROPERTIES LANGUAGE HIP
)

target_include_directories(mixbench-hip PRIVATE ../include)

target_link_libraries(mixbench-hip PRIVATE hip::host)
