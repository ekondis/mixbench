# mixbench
The purpose of this benchmark tool is to evaluate performance bounds of GPUs on mixed operational intensity kernels. The executed kernel is customized on a range of different operational intensity values. Modern GPUs are able to hide memory latency by switching execution to compute operations. Using this tool one can assess the practical optimum balance in both types of operations for a GPU. Both CUDA and OpenCL implementations have been developed.

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

Three executables will be produced: "mixbench-cuda", "mixbench-cuda-bs" and "mixbench-ocl". The two former comprise the CUDA implementations and the latter the OpenCL implementation. The fist applies grid strides between accesses of the same thread where the second applies block size strides. Both methods are supported by the OpenCL implementation and can be selected using a command line option.

Execution results
--------------

A typical execution output on an NVidia GTX660 GPU is:
```
mixbench (compute & memory balancing GPU microbenchmark)
------------------------ Device specifications ------------------------
Device:              GeForce GTX 660
CUDA driver version: 7.50
GPU clock rate:      1097 MHz
Memory clock rate:   1502 MHz
Memory bus width:    192 bits
WarpSize:            32
L2 cache size:       384 KB
Total global mem:    2042 MB
ECC enabled:         No
Compute Capability:  3.0
Total SPs:           960 (5 MPs x 192 SPs/MP)
Compute throughput:  2107.20 GFlops (theoretical single precision FMAs)
Memory bandwidth:    144.19 GB/sec
-----------------------------------------------------------------------
Total GPU memory 2141913088, free 2106126336
Buffer size: 256MB
Trade-off type:compute with global memory (block strided)
--------------------------------------------------- CSV data --------------------------------------------------
Single Precision ops,,,,              Double precision ops,,,,              Integer operations,,, 
Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec
     0.000,  331.11,    0.00, 103.77,      0.000,  666.67,    0.00, 103.08,     0.000,  330.91,    0.00, 103.83
     0.065,  321.70,    6.68, 103.47,      0.032,  645.48,    3.33, 103.14,     0.065,  321.53,    6.68, 103.53
     0.133,  310.38,   13.84, 103.78,      0.067,  623.54,    6.89, 103.32,     0.133,  310.32,   13.84, 103.80
     0.207,  299.76,   21.49, 103.88,      0.103,  603.28,   10.68, 103.23,     0.207,  299.81,   21.49, 103.86
     0.286,  288.37,   29.79, 104.26,      0.143,  581.11,   14.78, 103.47,     0.286,  288.19,   29.81, 104.32
     0.370,  278.62,   38.54, 104.05,      0.185,  561.75,   19.11, 103.22,     0.370,  279.04,   38.48, 103.90
     0.462,  264.36,   48.74, 105.60,      0.231,  540.07,   23.86, 103.38,     0.462,  264.69,   48.68, 105.47
     0.560,  254.86,   58.98, 105.33,      0.280,  520.83,   28.86, 103.08,     0.560,  255.16,   58.91, 105.20
     0.667,  242.78,   70.76, 106.14,      0.333,  497.84,   34.51, 103.53,     0.667,  243.07,   70.68, 106.02
     0.783,  231.60,   83.45, 106.63,      0.391,  478.86,   40.36, 103.14,     0.783,  232.46,   83.14, 106.24
     0.909,  217.54,   98.72, 108.59,      0.455,  456.43,   47.05, 103.51,     0.909,  217.62,   98.68, 108.55
     1.048,  208.25,  113.43, 108.28,      0.524,  435.83,   54.20, 103.47,     1.048,  208.34,  113.38, 108.23
     1.200,  194.99,  132.16, 110.14,      0.600,  413.28,   62.35, 103.92,     1.200,  195.24,  131.99, 109.99
     1.368,  186.92,  149.36, 109.14,      0.684,  392.24,   71.17, 104.02,     1.368,  185.82,  150.24, 109.79
     1.556,  173.37,  173.42, 111.48,      0.778,  356.28,   84.39, 108.50,     1.556,  172.02,  174.77, 112.35
     1.765,  164.31,  196.04, 111.09,      0.882,  378.13,   85.19,  96.55,     1.765,  161.31,  199.69, 113.16
     2.000,  166.16,  206.78, 103.39,      1.000,  401.90,   85.49,  85.49,     2.000,  165.62,  207.46, 103.73
     2.267,  150.74,  242.19, 106.85,      1.133,  425.10,   85.88,  75.78,     2.267,  151.01,  241.75, 106.65
     2.571,  144.55,  267.42, 104.00,      1.286,  449.01,   86.09,  66.96,     2.571,  143.85,  268.71, 104.50
     2.923,  130.06,  313.73, 107.33,      1.462,  470.90,   86.65,  59.29,     2.923,  132.64,  307.61, 105.23
     3.333,  122.40,  350.90, 105.27,      1.667,  492.46,   87.21,  52.33,     3.333,  135.07,  317.97,  95.39
     3.818,  108.68,  414.96, 108.68,      1.909,  515.52,   87.48,  45.82,     3.818,  135.69,  332.36,  87.05
     4.400,  103.54,  456.29, 103.70,      2.200,  540.59,   87.39,  39.72,     4.400,  143.56,  329.10,  74.79
     5.111,   86.99,  567.81, 111.09,      2.556,  562.18,   87.86,  34.38,     5.111,  148.07,  333.56,  65.26
     6.000,   82.94,  621.40, 103.57,      3.000,  586.40,   87.89,  29.30,     6.000,  157.30,  327.65,  54.61
     7.143,   66.18,  811.25, 113.58,      3.571,  612.78,   87.61,  24.53,     7.143,  162.70,  329.97,  46.20
     8.667,   61.83,  902.97, 104.19,      4.333,  636.91,   87.67,  20.23,     8.667,  170.39,  327.70,  37.81
    10.800,   44.39, 1306.09, 120.93,      5.400,  658.38,   88.07,  16.31,    10.800,  173.27,  334.64,  30.98
    14.000,   41.02, 1465.77, 104.70,      7.000,  681.96,   88.17,  12.60,    14.000,  177.82,  338.15,  24.15
    19.333,   39.30, 1584.64,  81.96,      9.667,  703.63,   88.51,   9.16,    19.333,  182.63,  340.99,  17.64
    30.000,   38.84, 1658.88,  55.30,     15.000,  724.53,   88.92,   5.93,    30.000,  189.26,  340.40,  11.35
    62.000,   35.55, 1872.54,  30.20,     31.000,  743.62,   89.52,   2.89,    62.000,  188.37,  353.41,   5.70
       inf,   35.92, 1912.98,   0.00,        inf,  765.54,   89.77,   0.00,       inf,  191.35,  359.14,   0.00
---------------------------------------------------------------------------------------------------------------
```

Publications
--------------

If you use this benchmark tool for a research work please provide citation to the following paper:

Konstantinidis, E.; Cotronis, Y., "A Practical Performance Model for Compute and Memory Bound GPU Kernels," Parallel, Distributed and Network-Based Processing (PDP), 2015 23rd Euromicro International Conference on , vol., no., pp.651,658, 4-6 March 2015
doi: 10.1109/PDP.2015.51
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7092788&isnumber=7092002
