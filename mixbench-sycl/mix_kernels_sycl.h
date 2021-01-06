/**
 * mix_kernels_sycl.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _MIX_KERNELS_SYCL_H_
#define _MIX_KERNELS_SYCL_H_

void mixbenchGPU(const sycl::device&, double*, long, bool, size_t);

#endif
