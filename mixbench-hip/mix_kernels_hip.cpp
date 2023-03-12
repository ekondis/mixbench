/**
 * mix_kernels_hip.cpp: This file is part of the mixbench GPU micro-benchmark
 *suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <iostream>
#ifdef __CUDACC__
#include <math_constants.h>
#define GPU_INF(_T) (_T)(CUDART_INF)
#else
#include <limits>
#define GPU_INF(_T) std::numeric_limits<_T>::infinity()
#endif

typedef __half2 half2;

#include <common.h>
#include "lhiputil.h"

#define ELEMENTS_PER_THREAD (8)

template <class T>
inline __device__ T mad(const T& a, const T& b, const T& c) {
  return a * b + c;
}

template <>
inline __device__ double mad(const double& a,
                             const double& b,
                             const double& c) {
  return fma(a, b, c);
}

template <>
inline __device__ half2 mad(const half2& a, const half2& b, const half2& c) {
  return __hfma2(a, b, c);
}

template <class T>
inline __device__ bool is_equal(const T& a, const T& b) {
  return a == b;
}

template <>
inline __device__ bool is_equal(const half2& a, const half2& b) {
  return __hbeq2(a, b);
}

template <class T,
          int blockSize,
          unsigned int granularity,
          unsigned int compute_iterations>
__global__ void benchmark_func(T seed, T* g_data) {
  const int stride = blockSize;
  const int idx = hipBlockIdx_x * blockSize * granularity + hipThreadIdx_x;

  T tmps[granularity];
#pragma unroll
  for (int j = 0; j < granularity; j++) {
    // Load elements (memory intensive part)
    tmps[j] = g_data[idx + j * stride];
    // Perform computations (compute intensive part)
    for (int i = 0; i < compute_iterations; i++) {
      tmps[j] = mad<T>(tmps[j], tmps[j], seed);
    }
  }
  // Multiply add reduction
  T sum = static_cast<T>(0);
#pragma unroll
  for (int j = 0; j < granularity; j += 2) {
    sum = mad<T>(tmps[j], tmps[j + 1], sum);
  }
  // Dummy code
  if (is_equal(sum, static_cast<T>(-1)))  // Designed so it never executes
    g_data[idx] = sum;
}

void initializeEvents_ext(hipEvent_t* start, hipEvent_t* stop) {
  HIP_SAFE_CALL(hipEventCreate(start));
  HIP_SAFE_CALL(hipEventCreate(stop));
}

float finalizeEvents_ext(hipEvent_t start, hipEvent_t stop) {
  HIP_SAFE_CALL(hipGetLastError());
  HIP_SAFE_CALL(hipEventSynchronize(stop));
  float kernel_time;
  HIP_SAFE_CALL(hipEventElapsedTime(&kernel_time, start, stop));
  HIP_SAFE_CALL(hipEventDestroy(start));
  HIP_SAFE_CALL(hipEventDestroy(stop));
  return kernel_time;
}

void runbench_warmup(double* cd, long size) {
  const long reduced_grid_size = size / (ELEMENTS_PER_THREAD) / 128;
  const int BLOCK_SIZE = 256;
  const int TOTAL_REDUCED_BLOCKS = reduced_grid_size / BLOCK_SIZE;

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          benchmark_func<short, BLOCK_SIZE, ELEMENTS_PER_THREAD, 0>),
      dim3(dimReducedGrid), dim3(dimBlock), 0, 0, (short)1, (short*)cd);
  HIP_SAFE_CALL(hipGetLastError());
  HIP_SAFE_CALL(hipDeviceSynchronize());
}

template <unsigned int compute_iterations>
void runbench(double* cd, long size) {
  const long compute_grid_size = size / ELEMENTS_PER_THREAD;
  const int BLOCK_SIZE = 256;
  const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;
  const long long computations =
      ELEMENTS_PER_THREAD * (long long)compute_grid_size +
      (2 * ELEMENTS_PER_THREAD * compute_iterations) *
          (long long)compute_grid_size;
  const long long memoryoperations = size;

  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
  hipEvent_t start, stop;

  constexpr auto total_bench_iterations = 3;

  float kernel_time_mad_sp = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start, &stop);
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<float, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start, stop, 0, 1.0f, (float*)cd);
    return finalizeEvents_ext(start, stop);
  });

  float kernel_time_mad_sp2 = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start, &stop);
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<float2, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start, stop, 0, float2{1.0f},
        (float2*)cd);
    return finalizeEvents_ext(start, stop);
  });

  float kernel_time_mad_dp = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start, &stop);
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<double, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start, stop, 0, 1.0, cd);
    return finalizeEvents_ext(start, stop);
  });

  float kernel_time_mad_hp = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start, &stop);
    half2 h_ones(1.0f);
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<half2, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start, stop, 0, h_ones,
        (half2*)cd);
    return finalizeEvents_ext(start, stop);
  });

  float kernel_time_mad_int = benchmark<total_bench_iterations>([&]() {
    initializeEvents_ext(&start, &stop);
    hipExtLaunchKernelGGL(
        HIP_KERNEL_NAME(benchmark_func<int, BLOCK_SIZE, ELEMENTS_PER_THREAD,
                                       compute_iterations>),
        dim3(dimGrid), dim3(dimBlock), 0, 0, start, stop, 0, 1, (int*)cd);
    return finalizeEvents_ext(start, stop);
  });

  printf(
      "         %4d,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,   "
      "%8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  "
      "%8.3f,%8.2f,%8.2f,%7.2f\n",
      compute_iterations,
      // SP
      ((double)computations) / ((double)memoryoperations * sizeof(float)),
      kernel_time_mad_sp,
      ((double)computations) / kernel_time_mad_sp * 1000. /
          (double)(1000 * 1000 * 1000),
      ((double)memoryoperations * sizeof(float)) / kernel_time_mad_sp * 1000. /
          (1000. * 1000. * 1000.),
      // Packed SP
      ((double)2 * computations) / ((double)memoryoperations * sizeof(float2)),
      kernel_time_mad_sp2,
      ((double)2 * computations) / kernel_time_mad_sp2 * 1000. /
          (double)(1000 * 1000 * 1000),
      ((double)memoryoperations * sizeof(float2)) / kernel_time_mad_sp2 *
          1000. / (1000. * 1000. * 1000.),
      // DP
      ((double)computations) / ((double)memoryoperations * sizeof(double)),
      kernel_time_mad_dp,
      ((double)computations) / kernel_time_mad_dp * 1000. /
          (double)(1000 * 1000 * 1000),
      ((double)memoryoperations * sizeof(double)) / kernel_time_mad_dp * 1000. /
          (1000. * 1000. * 1000.),
      // Packed HP
      ((double)2 * computations) / ((double)memoryoperations * sizeof(half2)),
      kernel_time_mad_hp,
      ((double)2 * computations) / kernel_time_mad_hp * 1000. /
          (double)(1000 * 1000 * 1000),
      ((double)memoryoperations * sizeof(half2)) / kernel_time_mad_hp * 1000. /
          (1000. * 1000. * 1000.),
      // Int
      ((double)computations) / ((double)memoryoperations * sizeof(int)),
      kernel_time_mad_int,
      ((double)computations) / kernel_time_mad_int * 1000. /
          (double)(1000 * 1000 * 1000),
      ((double)memoryoperations * sizeof(int)) / kernel_time_mad_int * 1000. /
          (1000. * 1000. * 1000.));
}

extern "C" void mixbenchGPU(double* c, long size) {
  const char* benchtype = "compute with global memory (block strided)";

  printf("Trade-off type:       %s\n", benchtype);
  printf("Elements per thread:  %d\n", ELEMENTS_PER_THREAD);
  printf("Thread fusion degree: %d\n", 1);
  double* cd;

  HIP_SAFE_CALL(hipMalloc((void**)&cd, size * sizeof(double)));

  // Copy data to device memory
  HIP_SAFE_CALL(
      hipMemset(cd, 0, size * sizeof(double)));  // initialize to zeros

  // Synchronize in order to wait for memory operations to finish
  HIP_SAFE_CALL(hipDeviceSynchronize());

  printf(
      "------------------------------------------------------------------------"
      "----- CSV data "
      "------------------------------------------------------------------------"
      "-------------------------------------------\n");
  printf(
      "Experiment ID, Single Precision ops,,,,              Packed Single "
      "Precision ops,,,,       Double precision ops,,,,              Half "
      "precision ops,,,,                Integer operations,,, \n");
  printf(
      "Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, "
      "ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, "
      "Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, "
      "GB/sec\n");

  runbench_warmup(cd, size);

  runbench<0>(cd, size);
  runbench<1>(cd, size);
  runbench<2>(cd, size);
  runbench<3>(cd, size);
  runbench<4>(cd, size);
  runbench<5>(cd, size);
  runbench<6>(cd, size);
  runbench<7>(cd, size);
  runbench<8>(cd, size);
  runbench<9>(cd, size);
  runbench<10>(cd, size);
  runbench<11>(cd, size);
  runbench<12>(cd, size);
  runbench<13>(cd, size);
  runbench<14>(cd, size);
  runbench<15>(cd, size);
  runbench<16>(cd, size);
  runbench<17>(cd, size);
  runbench<18>(cd, size);
  runbench<20>(cd, size);
  runbench<22>(cd, size);
  runbench<24>(cd, size);
  runbench<28>(cd, size);
  runbench<32>(cd, size);
  runbench<40>(cd, size);
  runbench<48>(cd, size);
  runbench<56>(cd, size);
  runbench<64>(cd, size);
  runbench<80>(cd, size);
  runbench<96>(cd, size);
  runbench<128>(cd, size);
  runbench<256>(cd, size);
  runbench<512>(cd, size);

  printf(
      "------------------------------------------------------------------------"
      "------------------------------------------------------------------------"
      "----------------------------------------------------------\n");

  // Copy results back to host memory
  HIP_SAFE_CALL(hipMemcpy(c, cd, size * sizeof(double), hipMemcpyDeviceToHost));

  HIP_SAFE_CALL(hipFree(cd));

  HIP_SAFE_CALL(hipDeviceReset());
}
