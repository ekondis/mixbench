/**
 * mix_kernels_cpu.cpp: This file is part of the mixbench GPU micro-benchmark
 *suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

const auto base_omp_get_max_threads = omp_get_max_threads();

using benchmark_clock = std::chrono::steady_clock;

#ifdef BASELINE_IMPL

template <typename Element, size_t compute_iterations, size_t static_chunk_size>
Element __attribute__((noinline)) bench_block(Element* data) {
  Element sum = 0;
  Element f = data[0];

#pragma omp simd aligned(data : 64) reduction(+ : sum)
  for (size_t i = 0; i < static_chunk_size; i++) {
    Element t = data[i];
    for (size_t j = 0; j < compute_iterations; j++) {
      t = t * t + f;
    }
    sum += t;
  }
  return sum;
}

#else

template <typename Element, size_t compute_iterations, size_t static_chunk_size>
Element __attribute__((noinline)) bench_block(Element* data) {
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

#endif

template <typename Element, size_t compute_iterations>
__attribute__((optimize("unroll-loops"))) size_t bench(size_t len,
                                                       const Element seed1,
                                                       const Element seed2,
                                                       Element* src) {
  Element sum = 0;
  constexpr size_t static_chunk_size = 4096;

#pragma omp parallel for reduction(+ : sum) schedule(static)
  for (size_t it_base = 0; it_base < len; it_base += static_chunk_size) {
    sum += bench_block<Element, compute_iterations, static_chunk_size>(
        &src[it_base]);
  }

  *src = sum;
  return len;
}

auto runbench_warmup(double* c, size_t size) {
  auto timer_start = benchmark_clock::now();

  bench<double, 16>(size, 1., -1., c);

  auto timer_duration = benchmark_clock::now() - timer_start;
  return std::chrono::duration_cast<std::chrono::microseconds>(timer_duration)
      .count();
}

template <typename Op>
auto measure_operation(Op op) {
  auto timer_start = benchmark_clock::now();
  op();
  auto timer_duration = benchmark_clock::now() - timer_start;
  return std::chrono::duration_cast<std::chrono::microseconds>(timer_duration)
             .count() /
         1000.;
}

template <typename Op>
auto benchmark_omp(Op op) {
  constexpr int total_runs = 20;
  constexpr int total_half_thread_runs = 10;

  auto duration = op();  // drop first measurement
  std::vector<decltype(duration)> measurements;

  // 1st try with full threading
  omp_set_num_threads(base_omp_get_max_threads);

  for (int i = 1; i < total_runs; i++) {
    duration = op();
    measurements.push_back(duration);
  }

  // then try with half threading
  if (base_omp_get_max_threads > 1) {
    omp_set_num_threads(base_omp_get_max_threads / 2);

    for (int i = 1; i < total_half_thread_runs; i++) {
      duration = op();
      measurements.push_back(duration);
    }
  }

  return *std::min_element(std::begin(measurements), std::end(measurements));
}

class ComputeSpace {
  size_t memory_space_{0};
  int compute_iterations_{0};

 public:
  ComputeSpace(size_t memory_space, int compute_iterations)
      : memory_space_{memory_space}, compute_iterations_{compute_iterations} {}

  template <typename T>
  size_t compute_ops() const {
    const auto total_elements = element_count<T>();
    const long long computations =
        total_elements            /* Vector length */
            * compute_iterations_ /* Core loop iteration count */
            * 2                   /* Flops per core loop iteration */
            * 1                   /* FMAs in the inner most loop */
        + total_elements - 1      /* Due to sum reduction */
        ;
    return computations;
  }

  size_t memory_traffic() const { return memory_space_; }

  template <typename T>
  size_t element_count() const {
    return memory_space_ / sizeof(T);
  }
};

template <unsigned int compute_iterations>
void runbench(double* c, size_t size) {
  ComputeSpace cs{size * sizeof(double), compute_iterations};

  // floating point part (single prec)
  auto kernel_time_mad_sp = benchmark_omp([&] {
    return measure_operation([&] {
      bench<float, compute_iterations>(cs.element_count<float>(), 1.f, -1.f,
                                       reinterpret_cast<float*>(c));
    });
  });

  // floating point part (double prec)
  auto kernel_time_mad_dp = benchmark_omp([&] {
    return measure_operation([&] {
      bench<double, compute_iterations>(cs.element_count<double>(), 1., -1., c);
    });
  });

  // integer part
  auto kernel_time_mad_int = benchmark_omp([&] {
    return measure_operation([&] {
      bench<int, compute_iterations>(cs.element_count<int>(), 1, -1,
                                     reinterpret_cast<int*>(c));
    });
  });

  const auto computations_sp = cs.compute_ops<float>();
  const auto computations_dp = cs.compute_ops<double>();
  const auto computations_int = cs.compute_ops<int>();
  const auto memory_traffic = cs.memory_traffic();

  const auto setw = std::setw;
  const auto setprecision = std::setprecision;
  std::cout << std::fixed << "         " << std::setw(4) << compute_iterations
            << ",   " << setw(8) << setprecision(3)
            << static_cast<double>(computations_sp) /
                   static_cast<double>(memory_traffic)
            << "," << setw(8) << setprecision(2) << kernel_time_mad_sp << ","
            << setw(8) << setprecision(2)
            << static_cast<double>(computations_sp) / kernel_time_mad_sp *
                   1000. / static_cast<double>(1000 * 1000 * 1000)
            << "," << setw(7) << setprecision(2)
            << static_cast<double>(memory_traffic) / kernel_time_mad_sp *
                   1000. / (1000. * 1000. * 1000.)

            << ",   " << setw(8) << setprecision(3)
            << static_cast<double>(computations_dp) /
                   static_cast<double>(memory_traffic)
            << "," << setw(8) << setprecision(2) << kernel_time_mad_dp << ","
            << setw(8) << setprecision(2)
            << static_cast<double>(computations_dp) / kernel_time_mad_dp *
                   1000. / static_cast<double>(1000 * 1000 * 1000)
            << "," << setw(7) << setprecision(2)
            << static_cast<double>(memory_traffic) / kernel_time_mad_dp *
                   1000. / (1000. * 1000. * 1000.)

            << ",  " << setw(8) << setprecision(3)
            << static_cast<double>(computations_int) /
                   static_cast<double>(memory_traffic)
            << "," << setw(8) << setprecision(2) << kernel_time_mad_int << ","
            << setw(8) << setprecision(2)
            << static_cast<double>(computations_int) / kernel_time_mad_int *
                   1000. / static_cast<double>(1000 * 1000 * 1000)
            << "," << setw(7) << setprecision(2)
            << static_cast<double>(memory_traffic) / kernel_time_mad_int *
                   1000. / (1000. * 1000. * 1000.)

            << std::endl;
}

// Variadic template helper to ease multiple configuration invocations
template <unsigned int compute_iterations>
void runbench_range(double* cd, long size) {
  runbench<compute_iterations>(cd, size);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range(double* cd, long size) {
  runbench_range<j1>(cd, size);
  runbench_range<j2, Args...>(cd, size);
}

void mixbenchCPU(double* c, size_t size) {
// Initialize data to zeros on memory by respecting 1st touch policy
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++)
    c[i] = 0.0;

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
