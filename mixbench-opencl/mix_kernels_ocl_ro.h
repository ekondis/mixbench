/**
 * mix_kernels_ocl_ro.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

#ifdef __APPLE__
#	include <OpenCL/OpenCL.h>
#else
#	include <CL/opencl.h>
#endif

extern "C" void mixbenchGPU(cl_device_id, double*, long, bool, bool, bool, size_t, unsigned int, unsigned int);

