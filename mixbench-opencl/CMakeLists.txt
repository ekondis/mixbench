cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(mixbench-ocl LANGUAGES CXX)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIR} "../include")

add_executable(mixbench-ocl-alt main-ocl.cpp mix_kernels_ocl.h loclutil.h mix_kernels_ocl.cpp mix_kernels_ocl.h loclutil.h)
add_executable(mixbench-ocl-ro main-ocl.cpp mix_kernels_ocl.h loclutil.h mix_kernels_ocl_ro.cpp mix_kernels_ocl_ro.h loclutil.h)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mix_kernels.cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mix_kernels_ro.cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_compile_definitions(mixbench-ocl-ro PRIVATE READONLY)

try_compile(HAVE_HALF2 ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/check-half2-def.cpp
		CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OpenCL_INCLUDE_DIR}")

#if(${OpenCL_VERSION_MAJOR} LESS 2)
if(NOT ${HAVE_HALF2})
	message( "cl_half2 workaround path" )
	target_compile_definitions(mixbench-ocl-alt PRIVATE HF_WORKAROUND)
	target_compile_definitions(mixbench-ocl-ro PRIVATE HF_WORKAROUND)
endif()

target_link_libraries(mixbench-ocl-alt ${OpenCL_LIBRARY})
target_link_libraries(mixbench-ocl-ro ${OpenCL_LIBRARY})
