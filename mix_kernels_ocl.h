/**
 * mix_kernels_ocl.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

#include <CL/opencl.h>

extern "C" void mixbenchGPU(cl_device_id, double*, long, bool, bool, size_t);

