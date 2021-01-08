# mixbench
The purpose of this benchmark tool is to evaluate performance bounds of GPUs on mixed operational intensity kernels. The executed kernel is customized on a range of different operational intensity values. Modern GPUs are able to hide memory latency by switching execution to threads able to perform compute operations. Using this tool one can assess the practical optimum balance in both types of operations for a GPU. CUDA, HIP, OpenCL and SYCL implementations have been developed.

Kernel types
--------------

Four types of experiments are executed combined with global memory accesses:

1. Single precision Flops (multiply-additions)
2. Double precision Flops (multiply-additions)
3. Half precision Flops (multiply-additions)
4. Integer multiply-addition operations

How to build
--------------

Building is based now on CMake files. Each implementation resides in a separate folder:

* CUDA implementation: `mixbench-cuda`
* OpenCL implementation: `mixbench-opencl`
* HIP implementation: `mixbench-hip`
* SYCL implementation: `mixbench-sycl`

Thus, to build a particular implementation use the proper `CMakeLists.txt`, e.g. for the OpenCL implementation you may use the commands as follows:

```
mkdir build
cd build
cmake ../mixbench-opencl
```

In some cases (depending on the CMake version) the OpenCL files might not be discovered automatically. In such cases you might need to provide the OpenCL directories explicitly, as in the examples below:

```
cmake ../mixbench-opencl -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
cmake ../mixbench-opencl -DOpenCL_LIBRARY=/opt/rocm/lib/libOpenCL.so -DOpenCL_INCLUDE_DIR=/opt/rocm/opencl/include/
cmake ../mixbench-opencl -DOpenCL_LIBRARY=/opt/amdgpu-pro/lib/x86_64-linux-gnu/libOpenCL.so
```

For HIP version, the HIP_PATH environment variable should be set to point to HIP installation directory. For more information follow the instructions on the following blog to properly install ROCK and ROCR drivers:  
* ROCm:  
https://github.com/RadeonOpenCompute/ROCm
* HIP:  
https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP

For the SYCL version, some example cmake invocations follow depending on the underlying platform:

* Intel clang/DPCPP (`per_kernel` mode facilitates cases where the device misses support for computations on a particular data type, e.g. double precision):
```
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_CXX_FLAGS="-fsycl -std=c++17 -fsycl-device-code-split=per_kernel"
# or ...
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=dpcpp -D CMAKE_CXX_FLAGS="-fsycl-device-code-split=per_kernel"
```

* AMD hipSYCL under ROCm (here building for two device architectures, *gfx803* & *gfx1012*):

```
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=syclcc -D CMAKE_CXX_FLAGS="--hipsycl-targets='omp;hip:gfx803,gfx1012' --rocm-device-lib-path=/opt/rocm/amdgcn/bitcode"
```

* NVidia clang/DPCPP
```
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_CXX_FLAGS="-fsycl -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
```

Two executables will be produced for each platform, `mixbench-XXX-alt` & `mixbench-XXX-ro`.
The first one (-alt) follows different design approach than the second one (-ro) so results typically sightly differ.
The one that exhibits better performance is dependent on the underlying architecture and compiler characteristics.

Execution results
--------------

A typical execution output on an NVidia RTX-2070 GPU is:
```
mixbench/read-only (v0.03-2-gbccfd71)
------------------------ Device specifications ------------------------
Device:              GeForce RTX 2070
CUDA driver version: 10.20
GPU clock rate:      1620 MHz
Memory clock rate:   3500 MHz
Memory bus width:    256 bits
WarpSize:            32
L2 cache size:       4096 KB
Total global mem:    7979 MB
ECC enabled:         No
Compute Capability:  7.5
Total SPs:           2304 (36 MPs x 64 SPs/MP)
Compute throughput:  7464.96 GFlops (theoretical single precision FMAs)
Memory bandwidth:    448.06 GB/sec
-----------------------------------------------------------------------
Total GPU memory 8366784512, free 7941521408
Buffer size:          256MB
Trade-off type:       compute with global memory (block strided)
Elements per thread:  8
Thread fusion degree: 4
----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------
Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Half precision ops,,,,                Integer operations,,, 
Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec
            0,      0.250,    0.32,  104.42, 417.68,      0.125,    0.63,   53.04, 424.35,      0.500,    0.32,  211.41, 422.81,     0.250,    0.32,  105.58, 422.30
            1,      0.750,    0.32,  316.34, 421.79,      0.375,    0.63,  158.69, 423.18,      1.500,    0.32,  634.22, 422.81,     0.750,    0.32,  317.30, 423.07
            2,      1.250,    0.32,  528.46, 422.77,      0.625,    0.78,  215.91, 345.45,      2.500,    0.32, 1055.97, 422.39,     1.250,    0.32,  528.57, 422.86
            3,      1.750,    0.32,  738.81, 422.17,      0.875,    1.08,  218.17, 249.34,      3.500,    0.32, 1478.95, 422.56,     1.750,    0.32,  740.59, 423.20
            4,      2.250,    0.32,  951.33, 422.81,      1.125,    1.38,  219.57, 195.17,      4.500,    0.32, 1902.66, 422.81,     2.250,    0.32,  950.66, 422.51
            5,      2.750,    0.32, 1162.74, 422.81,      1.375,    1.67,  220.38, 160.28,      5.500,    0.32, 2328.52, 423.37,     2.750,    0.32, 1162.74, 422.81
            6,      3.250,    0.32, 1374.56, 422.94,      1.625,    1.97,  220.99, 135.99,      6.500,    0.32, 2756.62, 424.10,     3.250,    0.32, 1375.81, 423.32
            7,      3.750,    0.32, 1592.45, 424.65,      1.875,    2.27,  221.38, 118.07,      7.500,    0.32, 3169.50, 422.60,     3.750,    0.32, 1585.55, 422.81
            8,      4.250,    0.32, 1796.95, 422.81,      2.125,    2.57,  221.71, 104.33,      8.500,    0.32, 3587.76, 422.09,     4.250,    0.37, 1545.63, 363.68
            9,      4.750,    0.32, 2006.34, 422.39,      2.375,    2.87,  221.85,  93.41,      9.500,    0.32, 3995.38, 420.57,     4.750,    0.32, 1998.29, 420.69
           10,      5.250,    0.32, 2209.52, 420.86,      2.625,    3.17,  222.02,  84.58,     10.500,    0.32, 4439.54, 422.81,     5.250,    0.32, 2220.44, 422.94
           11,      5.750,    0.32, 2434.12, 423.32,      2.875,    3.47,  222.17,  77.28,     11.500,    0.32, 4855.01, 422.17,     5.750,    0.32, 2426.77, 422.05
           12,      6.250,    0.32, 2638.06, 422.09,      3.125,    3.78,  222.18,  71.10,     12.500,    0.32, 5227.20, 418.18,     6.250,    0.38, 2202.15, 352.34
           13,      6.750,    0.32, 2841.95, 421.03,      3.375,    4.08,  222.30,  65.87,     13.500,    0.32, 5712.58, 423.15,     6.750,    0.32, 2850.54, 422.30
           14,      7.250,    0.32, 3065.39, 422.81,      3.625,    4.37,  222.45,  61.36,     14.500,    0.32, 6135.74, 423.15,     7.250,    0.32, 3065.08, 422.77
           15,      7.750,    0.33, 3143.40, 405.60,      3.875,    4.67,  222.57,  57.44,     15.500,    0.32, 6546.34, 422.34,     7.750,    0.32, 3268.89, 421.79
           16,      8.250,    0.32, 3482.59, 422.13,      4.125,    4.98,  222.57,  53.96,     16.500,    0.32, 6957.48, 421.67,     8.250,    0.39, 2803.68, 339.84
           17,      8.750,    0.32, 3693.66, 422.13,      4.375,    5.28,  222.53,  50.86,     17.500,    0.32, 7396.24, 422.64,     8.750,    0.32, 3694.77, 422.26
           18,      9.250,    0.32, 3901.58, 421.79,      4.625,    5.58,  222.58,  48.12,     18.500,    0.32, 7786.72, 420.90,     9.250,    0.32, 3897.66, 421.37
           20,     10.250,    0.32, 4312.53, 420.73,      5.125,    6.18,  222.66,  43.45,     20.500,    0.32, 8640.66, 421.50,    10.250,    0.41, 3374.54, 329.22
           22,     11.250,    0.32, 4729.94, 420.44,      5.625,    6.78,  222.74,  39.60,     22.500,    0.32, 9452.31, 420.10,    11.250,    0.32, 4734.21, 420.82
           24,     12.250,    0.32, 5148.83, 420.31,      6.125,    7.36,  223.51,  36.49,     24.500,    0.32,10346.40, 422.30,    12.250,    0.42, 3900.12, 318.38
           28,     14.250,    0.32, 6009.94, 421.75,      7.125,    8.53,  224.23,  31.47,     28.500,    0.32,11975.32, 420.19,    14.250,    0.44, 4368.11, 306.53
           32,     16.250,    0.32, 6795.36, 418.18,      8.125,    9.72,  224.31,  27.61,     32.500,    0.32,13605.64, 418.64,    16.250,    0.45, 4797.12, 295.21
           40,     20.250,    0.34, 7899.43, 390.10,     10.125,   12.11,  224.50,  22.17,     40.500,    0.33,16371.37, 404.23,    20.250,    0.50, 5464.85, 269.87
           48,     24.250,    0.41, 8029.04, 331.09,     12.125,   14.49,  224.58,  18.52,     48.500,    0.40,16468.89, 339.56,    24.250,    0.54, 5986.22, 246.85
           56,     28.250,    0.47, 8114.58, 287.24,     14.125,   16.88,  224.65,  15.90,     56.500,    0.46,16443.12, 291.03,    28.250,    0.60, 6342.42, 224.51
           64,     32.250,    0.53, 8154.47, 252.85,     16.125,   19.26,  224.72,  13.94,     64.500,    0.52,16536.22, 256.38,    32.250,    0.66, 6591.93, 204.40
           80,     40.250,    0.66, 8242.80, 204.79,     20.125,   24.03,  224.79,  11.17,     80.500,    0.65,16644.88, 206.77,    40.250,    0.78, 6909.54, 171.67
           96,     48.250,    0.78, 8321.35, 172.46,     24.125,   28.80,  224.85,   9.32,     96.500,    0.78,16685.23, 172.90,    48.250,    0.91, 7108.62, 147.33
          128,     64.250,    1.03, 8337.22, 129.76,     32.125,   38.34,  224.91,   7.00,    128.500,    1.03,16775.65, 130.55,    64.250,    1.18, 7295.18, 113.54
          192,     96.250,    1.54, 8414.49,  87.42,     48.125,   57.42,  224.97,   4.67,    192.500,    1.53,16847.93,  87.52,    96.250,    1.74, 7431.64,  77.21
          256,    128.250,    2.06, 8362.01,  65.20,     64.125,   76.50,  225.02,   3.51,    256.500,    2.06,16693.65,  65.08,   128.250,    2.30, 7477.75,  58.31
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

And here is a chart illustrating the results extracted above:

![RTX-2070 execution results](https://raw.githubusercontent.com/ekondis/mixbench/gh-pages/img/rtx2070-sp-roofline.png "mixbench execution results on NVidia RTX-2070 (CUDA/ro implementation)")

Publications
--------------

If you use this benchmark tool for a research work please provide citation to any of the following papers:

Elias Konstantinidis, Yiannis Cotronis,
"A quantitative roofline model for GPU kernel performance estimation using micro-benchmarks and hardware metric profiling",
Journal of Parallel and Distributed Computing, Volume 107, September 2017, Pages 37-56, ISSN 0743-7315,
https://doi.org/10.1016/j.jpdc.2017.04.002.  
URL: http://www.sciencedirect.com/science/article/pii/S0743731517301247

Konstantinidis, E., Cotronis, Y.,
"A Practical Performance Model for Compute and Memory Bound GPU Kernels",
Parallel, Distributed and Network-Based Processing (PDP), 2015 23rd Euromicro International Conference on , vol., no., pp.651-658, 4-6 March 2015
doi: 10.1109/PDP.2015.51  
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7092788&isnumber=7092002
