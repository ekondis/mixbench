/**
 * mix_kernels_cpu.cpp: This file is part of the mixbench GPU micro-benchmark
 *suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <omp.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

#include "common.h"

template <typename Element, size_t compute_iterations, size_t static_chunk_size>
Element __attribute__((noinline)) bench_block(Element *data, Element seed) {
  Element sum = 0;

  Element f[] = {data[0], data[1], data[2], data[3],
                 data[4], data[5], data[6], data[7]};

#pragma omp simd aligned(data : 64) reduction(+ : sum)
  for (size_t i = 0; i < static_chunk_size; i++) {
    Element t[] = {data[i], data[i], data[i], data[i],
                   data[i], data[i], data[i], data[i]};
    for (size_t j = 0; j < compute_iterations / 8; j++) {
      t[0] = t[0] * t[0] + f[0];
      t[1] = t[1] * t[1] + f[1];
      t[2] = t[2] * t[2] + f[2];
      t[3] = t[3] * t[3] + f[3];
      t[4] = t[4] * t[4] + f[4];
      t[5] = t[5] * t[5] + f[5];
      t[6] = t[6] * t[6] + f[6];
      t[7] = t[7] * t[7] + f[7];
    }
    if constexpr (compute_iterations % 8 > 0) {
      t[0] = t[0] * t[0] + f[0];
    }
    if constexpr (compute_iterations % 8 > 1) {
      t[1] = t[1] * t[1] + f[1];
    }
    if constexpr (compute_iterations % 8 > 2) {
      t[2] = t[2] * t[2] + f[2];
    }
    if constexpr (compute_iterations % 8 > 3) {
      t[3] = t[3] * t[3] + f[3];
    }
    if constexpr (compute_iterations % 8 > 4) {
      t[4] = t[4] * t[4] + f[4];
    }
    if constexpr (compute_iterations % 8 > 5) {
      t[5] = t[5] * t[5] + f[5];
    }
    if constexpr (compute_iterations % 8 > 6) {
      t[6] = t[6] * t[6] + f[6];
    }
    sum += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
  }
  return sum;
}

template <typename Element, size_t compute_iterations>
__attribute__((optimize("unroll-loops"))) size_t bench(size_t len,
                                                       const Element seed1,
                                                       const Element seed2,
                                                       Element *src) {
  Element sum = 0;
  constexpr size_t static_chunk_size = 4096;
#pragma omp parallel reduction(+ : sum)
  {
    auto id = omp_get_thread_num();
    auto count = omp_get_num_threads();
    const size_t chunk_size = len / static_cast<size_t>(count);
    const size_t chunk_base = static_cast<size_t>(id) * chunk_size;

    if (true) {
      for (size_t it_base = chunk_base; it_base < chunk_base + chunk_size;
           it_base += static_chunk_size) {
        sum += bench_block<Element, compute_iterations, static_chunk_size>(
            &src[it_base], seed1);
      }
    } else {
      Element f[] = {src[0], src[1], src[2], src[3],
                     src[4], src[5], src[6], src[7]};
      for (size_t it_base = chunk_base; it_base < chunk_base + chunk_size;
           it_base += static_chunk_size) {
#pragma omp simd aligned(src : 32) reduction(+ : sum)
        for (size_t i = 0; i < static_chunk_size; i++) {
          Element t[] = {src[it_base + i], src[it_base + i], src[it_base + i],
                         src[it_base + i], src[it_base + i], src[it_base + i],
                         src[it_base + i], src[it_base + i]};

          for (size_t j = 0; j < compute_iterations; j++) {
            t[0] = t[0] * t[0] + f[0];
            t[1] = t[1] * t[1] + f[1];
            t[2] = t[2] * t[2] + f[2];
            t[3] = t[3] * t[3] + f[3];
            t[4] = t[4] * t[4] + f[4];
            t[5] = t[5] * t[5] + f[5];
            t[6] = t[6] * t[6] + f[6];
            t[7] = t[7] * t[7] + f[7];
          }
          sum += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
        }
      }
    }
  }
  *src = sum;
  return len;
}

auto runbench_warmup(double *c, size_t size) {
  auto timer_start = std::chrono::high_resolution_clock::now();

  bench<double, 16>(size, 1., -1., c);

  auto timer_duration = std::chrono::high_resolution_clock::now() - timer_start;
  return std::chrono::duration_cast<std::chrono::microseconds>(timer_duration)
      .count();
}

using half2 = float;

template <typename Op>
auto measure_operation(Op op) {
  auto timer_start = std::chrono::high_resolution_clock::now();
  op();
  auto timer_duration = std::chrono::high_resolution_clock::now() - timer_start;
  return std::chrono::duration_cast<std::chrono::microseconds>(timer_duration)
             .count() /
         1000.;
}

template <unsigned int compute_iterations>
void runbench(double *c, size_t size) {
  // floating point part (single prec)
  auto kernel_time_mad_sp = benchmark([&] {
    return measure_operation([&] {
      bench<float, compute_iterations>(size, 1.f, -1.f,
                                       reinterpret_cast<float *>(c));
    });
  });

  // floating point part (double prec)
  auto kernel_time_mad_dp = benchmark([&] {
    return measure_operation(
        [&] { bench<double, compute_iterations>(size, 1., -1., c); });
  });

  // integer part
  auto kernel_time_mad_int = benchmark([&] {
    return measure_operation([&] {
      bench<int, compute_iterations>(size * sizeof(double) / sizeof(int), 1, -1,
                                     reinterpret_cast<int *>(c));
    });
  });

  const long long computations =
      size                     /* Vector length */
          * compute_iterations /* Core loop iteration count */
          * 2                  /* Flops per core loop iteration */
          * 1                  /* FMAs in the inner most loop */
      + size - 1               /* Due to sum reduction */
      ;
  const long long memoryoperations = size;

  const auto setw = std::setw;
  const auto setprecision = std::setprecision;
  std::cout << std::fixed << "         " << std::setw(4) << compute_iterations
            << ",   " << setw(8) << setprecision(3)
            << ((double)computations) /
                   ((double)memoryoperations * sizeof(float))
            << "," << setw(8) << setprecision(2) << kernel_time_mad_sp << ","
            << setw(8) << setprecision(2)
            << ((double)computations) / kernel_time_mad_sp * 1000. /
                   (double)(1000 * 1000 * 1000)
            << "," << setw(7) << setprecision(2)
            << ((double)memoryoperations * sizeof(float)) / kernel_time_mad_sp *
                   1000. / (1000. * 1000. * 1000.)

            << ",   " << setw(8) << setprecision(3)
            << ((double)computations) /
                   ((double)memoryoperations * sizeof(double))
            << "," << setw(8) << setprecision(2) << kernel_time_mad_dp << ","
            << setw(8) << setprecision(2)
            << ((double)computations) / kernel_time_mad_dp * 1000. /
                   (double)(1000 * 1000 * 1000)
            << "," << setw(7) << setprecision(2)
            << ((double)memoryoperations * sizeof(double)) /
                   kernel_time_mad_dp * 1000. / (1000. * 1000. * 1000.)

            << ",  " << setw(8) << setprecision(3)
            << ((double)computations) / ((double)memoryoperations * sizeof(int))
            << "," << setw(8) << setprecision(2) << kernel_time_mad_int << ","
            << setw(8) << setprecision(2)
            << ((double)computations) / kernel_time_mad_int * 1000. /
                   (double)(1000 * 1000 * 1000)
            << "," << setw(7) << setprecision(2)
            << ((double)memoryoperations * sizeof(int)) / kernel_time_mad_int *
                   1000. / (1000. * 1000. * 1000.)

            << std::endl;
}

// Variadic template helper to ease multiple configuration invocations
template <unsigned int compute_iterations>
void runbench_range(double *cd, long size) {
  runbench<compute_iterations>(cd, size);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range(double *cd, long size) {
  runbench_range<j1>(cd, size);
  runbench_range<j2, Args...>(cd, size);
}

void mixbenchCPU(double *c, size_t size) {
  // Initialize data to zeros on device memory
  for (size_t i = 0; i < size; i++) c[i] = 0.0;

  std::cout << "--------------------------------------------"
               "-------------- CSV data "
               "--------------------------------------------"
               "--------------"
            << std::endl;
  std::cout << "Experiment ID, Single Precision ops,,,,              Double "
               "precision ops,,,,              Integer operations,,, "
            << std::endl;
  std::cout << "Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, "
               "Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   "
               "GIOPS, GB/sec"
            << std::endl;

  runbench_warmup(c, size);

  runbench_range<0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 6 * 8, 7 * 8,
                 8 * 8, 10 * 8, 13 * 8, 15 * 8, 16 * 8, 20 * 8, 24 * 8, 32 * 8,
                 40 * 8, 64 * 8>(c, size);

  std::cout << "---------------------------------------------------------------"
               "---------------------------------------------------------------"
            << std::endl;
}
