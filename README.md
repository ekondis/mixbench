# mixbench
The purpose of this benchmark tool is to evaluate performance bounds of GPUs on mixed operational intensity kernels. The executed kernel is customized on a range of different operational intensity values. Modern GPUs are able to hide memory latency by switching execution to threads able to perform compute operations. Using this tool one can assess the practical optimum balance in both types of operations for a GPU. CUDA, HIP and OpenCL implementations have been developed.

Kernel types
--------------

Three types of experiments are executed combined with global memory accesses:

1. Single precision Flops (multiply-additions)
2. Double precision Flops (multiply-additions)
3. Integer multiply-addition operations

Building program
--------------

In order to build the program you should make sure that the following variables in "Makefile" is set to the CUDA/OpenCL installation directory:

```
CUDA_INSTALL_PATH = /usr/local/cuda
OCL_INSTALL_PATH = /opt/AMDAPPSDK
CUDA_INC_PATH = ${CUDA_INSTALL_PATH}/include
CUDA_LIB_PATH = ${CUDA_INSTALL_PATH}/lib64
OCL_INC_PATH = ${OCL_INSTALL_PATH}/include
OCL_LIB_PATH = ${OCL_INSTALL_PATH}/lib/x86_64
```

*CUDA_INSTALL_PATH* is required to locate nvcc compiler, *CUDA_INC_PATH* should point to the CUDA headers include path, *CUDA_LIB_PATH* should point to the CUDA libraries, *OCL_INC_PATH* should point to the OpenCL header file and *OCL_LIB_PATH* should point to the OpenCL library. *OCL_INSTALL_PATH* is not required to point at any particular SDK as long as the header and library can be located.

Afterwards, just do make:

```
make
```

For HIP version, the HIP_PATH environment variable should be set to point to HIP installation directory. For more information follow the instructions on the following blog to properly install ROCK and ROCR drivers:  
http://gpuopen.com/getting-started-with-boltzmann-components-platforms-installation/  
Install the HCC compiler:  
https://bitbucket.org/multicoreware/hcc/wiki/Home  
Install HIP:  
https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP

Makefile checks if HIP is available in the system, and generate HIP binaries accordingly.

Two executables will be produced for each platform, i.e. "mixbench-cuda" & "mixbench-cuda-ro", "mixbench-ocl" & "mixbench-ocl-ro" and "mixbench-hip" & "mixbench-hip-ro". The first one follows different design approach than the second one so results typically differ. The one that exhibits better performance is dependent on the underlying architecture and compiler characteristics.

Execution results
--------------

A typical execution output on an NVidia GTX480 GPU is:
```
mixbench/read-only (compute & memory balancing GPU microbenchmark)
------------------------ Device specifications ------------------------
Device:              GeForce GTX 480
CUDA driver version: 8.0
GPU clock rate:      1550 MHz
Memory clock rate:   950 MHz
Memory bus width:    384 bits
WarpSize:            32
L2 cache size:       768 KB
Total global mem:    1530 MB
ECC enabled:         No
Compute Capability:  2.0
Total SPs:           480 (15 MPs x 32 SPs/MP)
Compute throughput:  1488.00 GFlops (theoretical single precision FMAs)
Memory bandwidth:    182.40 GB/sec
-----------------------------------------------------------------------
Total GPU memory 1605042176, free 1493491712
Buffer size: 256MB
Trade-off type:compute with global memory (block strided)
---------------------------------------------------------- CSV data ----------------------------------------------------------
Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Integer operations,,,
Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec
            0,     0.250,    0.82,   40.94, 163.76,      0.125,    1.65,   20.28, 162.21,     0.250,    0.82,   40.89, 163.55
            1,     0.750,    0.82,  122.92, 163.89,      0.375,    1.65,   60.95, 162.53,     0.750,    0.82,  122.93, 163.90
            2,     1.250,    0.82,  204.70, 163.76,      0.625,    1.65,  101.44, 162.30,     1.250,    0.82,  204.78, 163.83
            3,     1.750,    0.82,  286.56, 163.75,      0.875,    1.65,  141.93, 162.21,     1.750,    0.82,  287.21, 164.12
            4,     2.250,    0.82,  367.95, 163.53,      1.125,    1.74,  173.98, 154.65,     2.250,    0.82,  368.84, 163.93
            5,     2.750,    0.82,  450.16, 163.69,      1.375,    2.06,  179.29, 130.39,     2.750,    0.83,  446.77, 162.46
            6,     3.250,    0.82,  531.46, 163.53,      1.625,    2.42,  180.31, 110.96,     3.250,    0.84,  518.01, 159.39
            7,     3.750,    0.82,  612.06, 163.22,      1.875,    2.77,  181.42,  96.76,     3.750,    0.87,  581.29, 155.01
            8,     4.250,    0.82,  691.81, 162.78,      2.125,    3.14,  181.79,  85.55,     4.250,    0.88,  647.41, 152.33
            9,     4.750,    0.83,  770.56, 162.22,      2.375,    3.50,  182.25,  76.74,     4.750,    0.93,  683.16, 143.82
           10,     5.250,    0.83,  846.47, 161.23,      2.625,    3.86,  182.67,  69.59,     5.250,    1.01,  695.39, 132.45
           11,     5.750,    0.84,  915.16, 159.16,      2.875,    4.22,  183.02,  63.66,     5.750,    1.10,  699.84, 121.71
           12,     6.250,    0.85,  989.75, 158.36,      3.125,    4.60,  182.55,  58.42,     6.250,    1.19,  707.60, 113.22
           13,     6.750,    0.86, 1053.37, 156.06,      3.375,    4.96,  182.73,  54.14,     6.750,    1.27,  711.56, 105.42
           14,     7.250,    0.88, 1107.91, 152.81,      3.625,    5.30,  183.54,  50.63,     7.250,    1.36,  714.69,  98.58
           15,     7.750,    0.90, 1156.71, 149.25,      3.875,    5.66,  183.66,  47.40,     7.750,    1.45,  717.05,  92.52
           16,     8.250,    0.93, 1184.79, 143.61,      4.125,    6.03,  183.78,  44.55,     8.250,    1.54,  718.77,  87.12
           17,     8.750,    0.96, 1219.80, 139.41,      4.375,    6.38,  183.94,  42.04,     8.750,    1.64,  717.85,  82.04
           18,     9.250,    1.00, 1237.59, 133.79,      4.625,    6.75,  184.06,  39.80,     9.250,    1.72,  721.70,  78.02
           20,    10.250,    1.08, 1270.36, 123.94,      5.125,    7.47,  184.19,  35.94,    10.250,    1.91,  721.23,  70.36
           22,    11.250,    1.17, 1295.39, 115.15,      5.625,    8.19,  184.33,  32.77,    11.250,    2.08,  727.47,  64.66
           24,    12.250,    1.25, 1313.87, 107.25,      6.125,    8.91,  184.43,  30.11,    12.250,    2.26,  727.94,  59.42
           28,    14.250,    1.43, 1335.77,  93.74,      7.125,   10.36,  184.61,  25.91,    14.250,    2.63,  727.97,  51.09
           32,    16.250,    1.62, 1347.92,  82.95,      8.125,   11.81,  184.75,  22.74,    16.250,    2.99,  730.50,  44.95
           40,    20.250,    1.97, 1378.34,  68.07,     10.125,   14.70,  184.93,  18.26,    20.250,    3.72,  730.44,  36.07
           48,    24.250,    2.33, 1395.57,  57.55,     12.125,   17.59,  185.05,  15.26,    24.250,    4.43,  734.33,  30.28
           56,    28.250,    2.69, 1407.40,  49.82,     14.125,   20.48,  185.14,  13.11,    28.250,    5.16,  735.41,  26.03
           64,    32.250,    3.06, 1413.34,  43.82,     16.125,   23.37,  185.20,  11.49,    32.250,    5.88,  736.20,  22.83
           80,    40.250,    3.78, 1430.36,  35.54,     20.125,   29.16,  185.29,   9.21,    40.250,    7.32,  737.66,  18.33
           96,    48.250,    4.53, 1429.22,  29.62,     24.125,   34.94,  185.34,   7.68,    48.250,    8.77,  738.56,  15.31
          128,    64.250,    5.96, 1446.85,  22.52,     32.125,   46.51,  185.42,   5.77,    64.250,   11.66,  739.62,  11.51
          192,    96.250,    8.90, 1451.22,  15.08,     48.125,   69.64,  185.50,   3.85,    96.250,   17.44,  740.59,   7.69
          256,   128.250,   11.80, 1458.44,  11.37,     64.125,   92.77,  185.54,   2.89,   128.250,   23.23,  741.12,   5.78
------------------------------------------------------------------------------------------------------------------------------
```

And here is a chart with illustrating data extracted by mixbench:
![GTX-480 and GTX-660 execution results](http://users.uoa.gr/~ekondis/shared/mixbench-thumb.png "mixbench execution results on GTX-480 and GTX-660 (CUDA/ro implementation)")

Publications
--------------

If you use this benchmark tool for a research work please provide citation to the following paper:

Konstantinidis, E.; Cotronis, Y.,
"A Practical Performance Model for Compute and Memory Bound GPU Kernels,"
Parallel, Distributed and Network-Based Processing (PDP), 2015 23rd Euromicro International Conference on , vol., no., pp.651-658, 4-6 March 2015
doi: 10.1109/PDP.2015.51  
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7092788&isnumber=7092002
