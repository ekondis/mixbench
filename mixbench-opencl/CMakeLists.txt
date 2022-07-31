cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(mixbench-ocl LANGUAGES CXX)

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIR} "../include")

# Get version info from git tag
execute_process(COMMAND git describe --tags --always
                OUTPUT_VARIABLE GIT_REV
                ERROR_QUIET)

# Store version info
if ("${GIT_REV}" STREQUAL "")
    set(GIT_REV "")
endif()
string(STRIP "${GIT_REV}" GIT_REV)
file(WRITE "version_info.h" "#define VERSION_INFO \"")
file(APPEND "version_info.h" ${GIT_REV})
file(APPEND "version_info.h" "\"")

add_executable(mixbench-ocl main-ocl.cpp loclutil.h mix_kernels_ocl.cpp mix_kernels_ocl.h loclutil.h)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mix_kernels.cl
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

try_compile(HAVE_HALF2 ${CMAKE_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/check-half2-def.cpp
		CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${OpenCL_INCLUDE_DIR}")

if(NOT ${HAVE_HALF2})
	message( "cl_half2 workaround path" )
	target_compile_definitions(mixbench-ocl PRIVATE HF_WORKAROUND)
endif()

target_link_libraries(mixbench-ocl ${OpenCL_LIBRARY})
