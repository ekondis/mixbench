/**
 * mix_kernels_sycl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <common.h>
#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "lsyclutil.h"

#define ELEMENTS_PER_THREAD (8)
#define FUSION_DEGREE (4)

#ifdef __HIPSYCL__
#include <hip/hip_fp16.h>
#else
using half2 = sycl::half2;
using half = sycl::half;
#endif

template <typename T, typename Enable = void>
struct MADOperator {
  T operator()(T a, T b, T c) { return a * b + c; }
};

#ifndef __HIPSYCL__
// Use partial specialization for calling sycl::mad() for generic floating point types
template <typename T>
struct MADOperator<T, typename std::enable_if_t<sycl::detail::is_genfloat<T>::value>> {
    T operator()(T a, T b, T c) {
        return sycl::mad(a, b, c);
        //return cl::sycl::fma(a, b, c);
    }
};
#else
#ifdef SYCL_DEVICE_ONLY
// Packed half precision operation support via ROCm
//
template <>
struct MADOperator<half2, void> {
    half2 operator()(half2 a, half2 b, half2 c) {
        return __hfma2(a, b, c);
    }
};
#endif
#endif

template <typename T>
struct EqualOperator {
    bool operator()(T a, T b) {
        return a == b;
    }
};

template <>
struct EqualOperator<half2> {
    bool operator()(half2 a, half2 b) {
#ifdef __HIPSYCL__
        return __hbeq2(a, b);
#else
        return a[0] == b[0] && a[1] == b[1];
#endif
    }
};

template <typename T>
struct FromIntOperator {
    T operator()(const int i) {
        return static_cast<T>(i);
    }
};

template <>
struct FromIntOperator<half2> {
    half2 operator()(const int i) {
        #ifdef __HIPSYCL__
        return half2{i,i};
        #else
        return sycl::int2{i}.convert<half, sycl::rounding_mode::rtn>();
        #endif
    }
};

template <class T, unsigned int granularity, unsigned int fusion_degree, unsigned int compute_iterations>
void benchmark_func(T seed, T *g_data, sycl::nd_item<1> item_ct1) {
    const unsigned int blockSize = item_ct1.get_local_range(0);
    const int stride = blockSize;
    int idx = item_ct1.get_group(0) * blockSize * granularity + item_ct1.get_local_id(0);
    const int big_stride = item_ct1.get_group_range(0) * blockSize * granularity;
    /*
#ifdef BLOCK_STRIDED
    const int stride = blockSize;
    const int idx = get_group_id(0)*blockSize*ELEMENTS_PER_THREAD + get_local_id(0);
#else
    const int grid_size = blockSize * get_num_groups(0);
    const int stride = grid_size;
    const int idx = get_global_id(0);
#endif
    const int big_stride = get_num_groups(0)*blockSize*ELEMENTS_PER_THREAD;
*/
    // Type specialized functors
    MADOperator<T> mad_op;
    EqualOperator<T> equal_op;
    FromIntOperator<T> from_int_op;
    T tmps[granularity];
    for (int k = 0; k < fusion_degree; k++) {
#pragma unroll
        for (int j = 0; j < granularity; j++) {
            // Load elements (memory intensive part)
            tmps[j] = g_data[idx + j * stride + k * big_stride];
            // Perform computations (compute intensive part)
            for (int i = 0; i < compute_iterations; i++) {
                tmps[j] = mad_op(tmps[j], tmps[j], seed);
            }
        }
        // Multiply add reduction
        T sum = from_int_op(0);
        //#pragma unroll
        for (int j = 0; j < granularity; j += 2) {
            sum = mad_op(tmps[j], tmps[j + 1], sum);
        }
        // Dummy code just to avoid dead code elimination
        if (equal_op(sum, from_int_op(-1))) {  // Designed so it never executes
            g_data[idx + k * big_stride] = sum;
        }
    }
}

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

time_point initializeEvents(void) {
    return std::chrono::high_resolution_clock::now();
}

double finalizeEvents(bool use_host_timer, sycl::event ev_krn_execution, const time_point &tp_start_compute) {
    ev_krn_execution.wait();
    if (use_host_timer) {
        const time_point tp_stop_compute = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(tp_stop_compute - tp_start_compute).count();
    } else {
        // Disabled for hipSYCL: error: no matching member function for call to 'get_profiling_info'
        return (ev_krn_execution.get_profiling_info<sycl::info::event_profiling::command_end>() -
                ev_krn_execution.get_profiling_info<sycl::info::event_profiling::command_start>()) /
               1000000.0;
    }
}

void runbench_warmup(sycl::queue &queue, void *cd, long size) {
    const long reduced_grid_size = size / (ELEMENTS_PER_THREAD) / 128;
    const int BLOCK_SIZE = 256;
    const int TOTAL_REDUCED_BLOCKS = reduced_grid_size / BLOCK_SIZE;

    sycl::range<1> dimBlock(BLOCK_SIZE);
    sycl::range<1> dimReducedGrid(TOTAL_REDUCED_BLOCKS);

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class krn_short>(
            sycl::nd_range<1>(dimReducedGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<1> item_ct1) {
                benchmark_func<short, ELEMENTS_PER_THREAD, FUSION_DEGREE, 0>(
                    (short)1, (short *)cd, item_ct1);
            });
    });

    queue.wait();
}

// forward declarations of kernel classes
template <unsigned int>
class krn_float;
template <unsigned int>
class krn_double;
template <unsigned int>
class krn_half;
template <unsigned int>
class krn_int;

template <unsigned int compute_iterations>
void runbench(sycl::queue &queue, void *cd, long size, bool doHalfs, bool doDoubles, bool use_os_timer, size_t workgroupsize) {
    const long compute_grid_size = size / ELEMENTS_PER_THREAD / FUSION_DEGREE;
    const int BLOCK_SIZE = workgroupsize;
    const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;

    const sycl::range<1> dimBlock{static_cast<unsigned long>(BLOCK_SIZE)};
    const sycl::range<1> dimGrid{static_cast<unsigned long>(TOTAL_BLOCKS)};

    constexpr auto total_bench_iterations = 3;

    // floating point part (single prec)
    auto kernel_time_mad_sp = benchmark<total_bench_iterations>([&]() {
      time_point tp_start_compute = initializeEvents();
      auto ev_exec = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class krn_float<compute_iterations>>(
            sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<1> item_ct1) {
              benchmark_func<float, ELEMENTS_PER_THREAD, FUSION_DEGREE,
                             compute_iterations>(-1.0f, (float*)cd, item_ct1);
            });
      });
      return finalizeEvents(use_os_timer, ev_exec, tp_start_compute);
    });

    // floating point part (double prec)
    double kernel_time_mad_dp = 0.;
    if (doDoubles) {
      kernel_time_mad_dp = benchmark<total_bench_iterations>([&]() {
        time_point tp_start_compute = initializeEvents();
        auto ev_exec = queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for<class krn_double<compute_iterations>>(
              sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<1> item_ct1) {
                benchmark_func<double, ELEMENTS_PER_THREAD, FUSION_DEGREE,
                               compute_iterations>(-1.0, reinterpret_cast<double*>(cd), item_ct1);
              });
        });
        return finalizeEvents(use_os_timer, ev_exec, tp_start_compute);
      });
    }

    double kernel_time_mad_hp = 0.;
    // floating point part (half prec)
    if (doHalfs) {
      kernel_time_mad_hp = benchmark<total_bench_iterations>([&]() {
        time_point tp_start_compute = initializeEvents();
        half2 h_ones{-1.0f, -1.0f};
        auto ev_exec = queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for<class krn_half<compute_iterations>>(
              sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
              [=](sycl::nd_item<1> item_ct1) {
                benchmark_func<half2, ELEMENTS_PER_THREAD, FUSION_DEGREE,
                               compute_iterations>(
                    h_ones, reinterpret_cast<half2*>(cd), item_ct1);
              });
        });
        return finalizeEvents(use_os_timer, ev_exec, tp_start_compute);
      });
    }

    // integer part
    auto kernel_time_mad_int = benchmark<total_bench_iterations>([&]() {
      time_point tp_start_compute = initializeEvents();
      auto ev_exec = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class krn_int<compute_iterations>>(
            sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<1> item_ct1) {
              benchmark_func<int, ELEMENTS_PER_THREAD, FUSION_DEGREE,
                             compute_iterations>(
                  -1, (int*)cd, item_ct1);  // seed 1 causes unwanted code
                                            // elimination optimization
            });
      });
      return finalizeEvents(use_os_timer, ev_exec, tp_start_compute);
    });

    const long long computations = (ELEMENTS_PER_THREAD * (long long)compute_grid_size + (2 * ELEMENTS_PER_THREAD * compute_iterations) * (long long)compute_grid_size) * FUSION_DEGREE;
    const long long memoryoperations = size;

    const auto setw = std::setw;
    const auto setprecision = std::setprecision;
    std::cout << std::fixed << "         " << std::setw(4) << compute_iterations
              << ",   " << setw(8) << setprecision(3) << ((double)computations) / ((double)memoryoperations * sizeof(float))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_sp
              << "," << setw(8) << setprecision(2) << ((double)computations) / kernel_time_mad_sp * 1000. / (double)(1000 * 1000 * 1000)
              << "," << setw(7) << setprecision(2) << ((double)memoryoperations * sizeof(float)) / kernel_time_mad_sp * 1000. / (1000. * 1000. * 1000.)

              << ",   " << setw(8) << setprecision(3) << ((double)computations) / ((double)memoryoperations * sizeof(double))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_dp
              << "," << setw(8) << setprecision(2) << ((double)computations) / kernel_time_mad_dp * 1000. / (double)(1000 * 1000 * 1000)
              << "," << setw(7) << setprecision(2) << ((double)memoryoperations * sizeof(double)) / kernel_time_mad_dp * 1000. / (1000. * 1000. * 1000.)

              << ",   " << setw(8) << setprecision(3) << ((double)2 * computations) / ((double)memoryoperations * sizeof(half2))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_hp
              << "," << setw(8) << setprecision(2) << ((double)2 * computations) / kernel_time_mad_hp * 1000. / (double)(1000 * 1000 * 1000)
              << "," << setw(7) << setprecision(2) << ((double)memoryoperations * sizeof(half2)) / kernel_time_mad_hp * 1000. / (1000. * 1000. * 1000.)

              << ",  " << setw(8) << setprecision(3) << ((double)computations) / ((double)memoryoperations * sizeof(int))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_int
              << "," << setw(8) << setprecision(2) << ((double)computations) / kernel_time_mad_int * 1000. / (double)(1000 * 1000 * 1000)
              << "," << setw(7) << setprecision(2) << ((double)memoryoperations * sizeof(int)) / kernel_time_mad_int * 1000. / (1000. * 1000. * 1000.)

              << std::endl;
}

// Variadic template helper to ease multiple configuration invocations
template <unsigned int compute_iterations>
void runbench_range(sycl::queue &queue, void *cd, long size, bool doHalfs, bool doDoubles, bool use_os_timer, size_t workgroupsize) {
    runbench<compute_iterations>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range(sycl::queue &queue, void *cd, long size, bool doHalfs, bool doDoubles, bool use_os_timer, size_t workgroupsize) {
    runbench_range<j1>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);
    runbench_range<j2, Args...>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);
}

void mixbenchGPU(const sycl::device &dev, void *c, long size, bool use_os_timer, size_t workgroupsize) {
    const sycl::property_list queue_prop_list = use_os_timer ? sycl::property_list{} : sycl::property_list{sycl::property::queue::enable_profiling()};
    sycl::queue queue{dev, queue_prop_list};

    std::cout << "Elements per thread:  " << ELEMENTS_PER_THREAD << std::endl;
    std::cout << "Thread fusion degree: " << FUSION_DEGREE << std::endl;
    std::cout << "Timer:                " << (use_os_timer ? "OS based" : "SYCL event based") << std::endl;

#ifndef __HIPSYCL__
    const bool doHalfs = dev.has(sycl::aspect::fp16);
    if (!doHalfs) {
        std::cout << "Warning:              Half precision computations are not supported" << std::endl;
    }

    const bool doDoubles = dev.has(sycl::aspect::fp64);
    if (!doDoubles) {
        std::cout << "Warning:              Double precision computations are not supported" << std::endl;
    }
#else
    const bool doHalfs = true;
    const bool doDoubles = true;
    std::cout << "Warning:              hipSYCL - Assuming half and double precision support" << std::endl;
#endif

    double *cd = sycl::malloc_device<double>(size, queue);

    // Initialize data to zeros on device memory
    queue.memset(cd, 0, size * sizeof(double));

    // Synchronize in order to wait for memory operations to finish
    queue.wait();

    std::cout << "----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------" << std::endl;
    std::cout << "Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Half precision ops,,,,                Integer operations,,, " << std::endl;
    std::cout << "Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec" << std::endl;

    runbench_warmup(queue, cd, size);

    runbench_range<0, 1, 2, 3, 4, 5, 6, 7, 8,
                   9, 10, 11, 12, 13, 14, 15, 16,
                   17, 18, 20, 22, 24, 28, 32, 40,
                   48, 56, 64, 80, 96, 128, 192, 256>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);

    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    // Copy results to host memory and release device memory
    queue.memcpy(c, cd, size * sizeof(double)).wait();

    sycl::free(cd, queue);
}
