/**
 * mix_kernels_sycl_alt.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include "lsyclutil.h"

namespace sycl = cl::sycl;

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define REGBLOCK_SIZE (8)

#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)

#ifdef __HIPSYCL__
#include <hip/hip_fp16.h>
#else
using half2 = sycl::half2;
using half = sycl::half;
#endif

template <typename T, typename Enable = void>
struct MADOperator {
    T operator()(T a, T b, T c) {
        return a * b + c;
    }
};

#ifndef __HIPSYCL__
// Use partial specialization for calling sycl::mad() for generic floating point types
template <typename T>
struct MADOperator<T, typename std::enable_if_t<sycl::detail::is_genfloat<T>::value>> {
    T operator()(T a, T b, T c) {
        return sycl::mad(a, b, c);
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

template<class T, class T_Storage = T>
inline void volatile_set(volatile T_Storage &p, T v){ p = v; }

template<>
inline void volatile_set<half2, uint32_t>(volatile uint32_t &p, half2 v){
    static_assert(sizeof(half2)==sizeof(uint32_t));
    uint32_t v_as_int;
    std::memcpy(&v_as_int, &v, sizeof(half2));
    p = v_as_int;
// leads clang++ to core dump
//	uint32_t &v_as_int = reinterpret_cast<uint32_t&>(v);
//    p = v_as_int;	
}

template<class T, class T_Storage = T>
inline T volatile_get(volatile T_Storage &p){ return p; }

template<>
inline half2 volatile_get<half2, uint32_t>(volatile uint32_t &p) {
    static_assert(sizeof(half2)==sizeof(uint32_t));
    uint32_t v = p;
    half2 res;
    std::memcpy(&res, &v, sizeof(half2));
// leads clang++ to core dump
//    half2 res = reinterpret_cast<half2&>(v);
    return res;
}

template <class T, int memory_ratio, class T_Storage = T>
void benchmark_func(T seed, volatile T_Storage *g_data, sycl::nd_item<1> item_ct1){
    const auto blockdim = item_ct1.get_local_range(0);
    const int index_stride = blockdim;
    const int index_base = item_ct1.get_group(0) * blockdim * UNROLLED_MEMORY_ACCESSES + item_ct1.get_local_id(0);

    const int halfarraysize = item_ct1.get_group_range(0) * blockdim * UNROLLED_MEMORY_ACCESSES;
    const int offset_slips = 1+UNROLLED_MEMORY_ACCESSES-((memory_ratio+1)/2);
    const int array_index_bound = index_base+offset_slips*index_stride;
    const int initial_index_range = memory_ratio>0 ? UNROLLED_MEMORY_ACCESSES % ((memory_ratio+1)/2) : 1;
    int initial_index_factor = 0;

    int array_index = index_base;
    MADOperator<T> mad_op;
    EqualOperator<T> equal_op;
    FromIntOperator<T> from_int_op;
    T r0 = seed + from_int_op(item_ct1.get_group(0) * blockdim + item_ct1.get_local_id(0)),
      r1 = r0 +  from_int_op(2), r2 = r0 + from_int_op(3),
      r3 = r0 +  from_int_op(5), r4 = r0 + from_int_op(7),
      r5 = r0 + from_int_op(11), r6 = r0 + from_int_op(13),
      r7 = r0 + from_int_op(17);

    for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
        #pragma unroll
        for(int i=0; i<UNROLL_ITERATIONS-memory_ratio; i++){
            r0 = mad_op(r0, r0, r4);
            r1 = mad_op(r1, r1, r5);
            r2 = mad_op(r2, r2, r6);
            r3 = mad_op(r3, r3, r7);
            r4 = mad_op(r4, r4, r0);
            r5 = mad_op(r5, r5, r1);
            r6 = mad_op(r6, r6, r2);
            r7 = mad_op(r7, r7, r3);
        }
        bool do_write = true;
        int reg_idx = 0;
        #pragma unroll
        for(int i=UNROLL_ITERATIONS-memory_ratio; i<UNROLL_ITERATIONS; i++){
            // Each iteration maps to one memory operation
            T& r = reg_idx==0 ? r0 : (reg_idx==1 ? r1 : (reg_idx==2 ? r2 : (reg_idx==3 ? r3 : (reg_idx==4 ? r4 : (reg_idx==5 ? r5 : (reg_idx==6 ? r6 : r7))))));
            if( do_write )
                volatile_set(g_data[ array_index+halfarraysize ], r);
            else {
                r = volatile_get<T, T_Storage>(g_data[ array_index ]);
                if( ++reg_idx>=REGBLOCK_SIZE )
                    reg_idx = 0;
                array_index += index_stride;
            }
            do_write = !do_write;
        }
        if( array_index >= array_index_bound ){
            if( ++initial_index_factor > initial_index_range)
                initial_index_factor = 0;
            array_index = index_base + initial_index_factor*index_stride;
        }
    }
    const auto comp_val = from_int_op(-1);
    if( equal_op(r0, comp_val) && equal_op(r1, comp_val) && equal_op(r2, comp_val) && equal_op(r3, comp_val) &&
        equal_op(r4, comp_val) && equal_op(r5, comp_val) && equal_op(r6, comp_val) && equal_op(r7, comp_val) ) {
        // extremely unlikely to happen
        volatile_set(g_data[0], static_cast<T>((((r0+r1) + (r2+r3)) + ((r4+r5) + (r6+r7)))));
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
#ifndef __HIPSYCL__
        // Disabled for hipSYCL: error: no matching member function for call to 'get_profiling_info'
        return (ev_krn_execution.get_profiling_info<sycl::info::event_profiling::command_end>() -
                ev_krn_execution.get_profiling_info<sycl::info::event_profiling::command_start>()) /
               1000000.0;
#else
        return -1.0;
#endif
    }
}

void runbench_warmup(sycl::queue &queue, double *cd, long size) {
    const long reduced_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/32;
    const int BLOCK_SIZE = 256;
    const int TOTAL_REDUCED_BLOCKS = reduced_grid_size / BLOCK_SIZE;

    sycl::range<1> dimBlock(BLOCK_SIZE);
    sycl::range<1> dimReducedGrid(TOTAL_REDUCED_BLOCKS);

    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class krn_short>(
            sycl::nd_range<1>(dimReducedGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<1> item_ct1) {
                benchmark_func<short, 0>((short)1, (short *)cd, item_ct1);
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

template <unsigned int memory_ratio>
void runbench(sycl::queue &queue, double *cd, long size, bool doHalfs, bool doDoubles, bool use_os_timer, size_t workgroupsize) {
    static_assert(memory_ratio<=UNROLL_ITERATIONS, "ERROR: memory_ratio exceeds UNROLL_ITERATIONS");

    const long compute_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/2;
    const int BLOCK_SIZE = workgroupsize;
    const int TOTAL_BLOCKS = compute_grid_size / BLOCK_SIZE;

    const sycl::range<1> dimBlock{static_cast<unsigned long>(BLOCK_SIZE)};
    const sycl::range<1> dimGrid{static_cast<unsigned long>(TOTAL_BLOCKS)};

    // floating point part (single prec)
    time_point tp_start_compute = initializeEvents();
    auto ev_exec_sp = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class krn_float<memory_ratio>>(
            sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<1> item_ct1) {
                benchmark_func<float, memory_ratio>(-1.0f, (float *)cd, item_ct1);
            });
    });
    auto kernel_time_mad_sp = finalizeEvents(use_os_timer, ev_exec_sp, tp_start_compute);

    // floating point part (double prec)
    double kernel_time_mad_dp = 0.;
    if (doDoubles) {
        tp_start_compute = initializeEvents();
        auto ev_exec_dp = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class krn_double<memory_ratio>>(
                sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<1> item_ct1) {
                    benchmark_func<double, memory_ratio>(-1.0, cd, item_ct1);
                });
        });
        kernel_time_mad_dp = finalizeEvents(use_os_timer, ev_exec_dp, tp_start_compute);
    }

    double kernel_time_mad_hp = 0.;
    // floating point part (half prec)
    if (doHalfs) {
        tp_start_compute = initializeEvents();
        half2 h_ones{-1.0f, -1.0f};
        auto ev_exec_hp = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class krn_half<memory_ratio>>(
                sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
                [=](sycl::nd_item<1> item_ct1) {
                    benchmark_func<half2, memory_ratio, uint32_t>(h_ones, reinterpret_cast<uint32_t*>(cd), item_ct1);
                });
        });
        kernel_time_mad_hp = finalizeEvents(use_os_timer, ev_exec_hp, tp_start_compute);
    }

    // integer part
    tp_start_compute = initializeEvents();
    auto ev_exec_int = queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class krn_int<memory_ratio>>(
            sycl::nd_range<1>(dimGrid * dimBlock, dimBlock),
            [=](sycl::nd_item<1> item_ct1) {
                benchmark_func<int, memory_ratio>(-1, (int *)cd, item_ct1);   // seed 1 causes unwanted code elimination optimization
            });
    });
    auto kernel_time_mad_int = finalizeEvents(use_os_timer, ev_exec_int, tp_start_compute);

    const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*compute_grid_size;
    const long long memoryoperations = (long long)(COMP_ITERATIONS)*compute_grid_size;

    const double memaccesses_ratio = (double)(memory_ratio)/UNROLL_ITERATIONS;
    const double computations_ratio = 1.0-memaccesses_ratio;

    const auto setw = std::setw;
    const auto setprecision = std::setprecision;
    std::cout << std::fixed << "         " << std::setw(4) << UNROLL_ITERATIONS-memory_ratio
              << ",   " << setw(8) << setprecision(3) << (computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(float))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_sp
              << "," << setw(8) << setprecision(2) << (computations_ratio*(double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000)
              << "," << setw(7) << setprecision(2) << (memaccesses_ratio*(double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.)

              << ",   " << setw(8) << setprecision(3) << (computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(double))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_dp
              << "," << setw(8) << setprecision(2) << (computations_ratio*(double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000)
              << "," << setw(7) << setprecision(2) << (memaccesses_ratio*(double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.)

              << ",   " << setw(8) << setprecision(3) << (computations_ratio*(double)2*computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(half2))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_hp
              << "," << setw(8) << setprecision(2) << (computations_ratio*(double)2*computations)/kernel_time_mad_hp*1000./(double)(1000*1000*1000)
              << "," << setw(7) << setprecision(2) << (memaccesses_ratio*(double)memoryoperations*sizeof(half2))/kernel_time_mad_hp*1000./(1000.*1000.*1000.)

              << ",  " << setw(8) << setprecision(3) << (computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(int))
              << "," << setw(8) << setprecision(2) << kernel_time_mad_int
              << "," << setw(8) << setprecision(2) << (computations_ratio*(double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000)
              << "," << setw(7) << setprecision(2) << (memaccesses_ratio*(double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.)

              << std::endl;
}

// Variadic template helper to ease multiple configuration invocations
template <unsigned int memory_ratio>
void runbench_range(sycl::queue &queue, double *cd, long size, bool doHalfs, bool doDoubles, bool use_os_timer, size_t workgroupsize) {
    runbench<memory_ratio>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);
}

template <unsigned int j1, unsigned int j2, unsigned int... Args>
void runbench_range(sycl::queue &queue, double *cd, long size, bool doHalfs, bool doDoubles, bool use_os_timer, size_t workgroupsize) {
    runbench_range<j1>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);
    runbench_range<j2, Args...>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);
}

void mixbenchGPU(const sycl::device &dev, double *c, long size, bool use_os_timer, size_t workgroupsize) {
#ifndef __HIPSYCL__
    const sycl::property_list queue_prop_list = use_os_timer ? sycl::property_list{} : sycl::property_list{sycl::property::queue::enable_profiling()};
    sycl::queue queue{dev, queue_prop_list};
#else
    sycl::queue queue{dev};
#endif

#ifdef __HIPSYCL__
    use_os_timer = true;  // force OS timer on hipSYCL
    std::cout << "Warning:              Running under hipSYCL. SYCL profiling not supported" << std::endl;
#endif

    std::cout << "Timer:                " << (use_os_timer ? "OS based" : "SYCL event based") << std::endl;

#ifndef __HIPSYCL__
    const bool doHalfs = dev.has_extension("cl_khr_fp16");
    if (!doHalfs) {
        std::cout << "Warning:              Half precision computations are not supported" << std::endl;
    }

    const bool doDoubles = dev.has_extension("cl_khr_fp64");
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

    runbench_range<32, 31, 30, 29, 28, 27, 26, 25,
                   24, 23, 22, 21, 20, 19, 18, 17,
                   16, 15, 14, 13, 12, 11, 10,  9,
                    8, 7, 6, 5, 4, 3, 2, 1, 0>(queue, cd, size, doHalfs, doDoubles, use_os_timer, workgroupsize);

    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    // Copy results to host memory and release device memory
    queue.memcpy(c, cd, size * sizeof(double)).wait();

    sycl::free(cd, queue);
}
