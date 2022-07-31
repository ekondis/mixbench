cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mixbench LANGUAGES CXX CUDA)

# Include CUDA header directory in cpp files
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
string(APPEND CMAKE_CUDA_FLAGS " --cudart=static")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=[sm_52,compute_52] -gencode arch=compute_61,code=compute_61")

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

add_executable(mixbench-cuda main-cuda.cpp mix_kernels_cuda.h lcutil.h version_info.h mix_kernels_cuda.cu mix_kernels_cuda.h lcutil.h)

target_compile_features(mixbench-cuda PUBLIC cxx_std_11)

