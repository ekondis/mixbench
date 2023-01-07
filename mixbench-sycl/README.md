# mixbench-sycl

This is the SYCL implementation of mixbench.
As SYCL is supported by multiple implementations, not all of them have been tested.

## Building notes

### Intel clang/DPCPP

Using the latest version of OneAPI toolkit from Intel, you may try building as follows:

```
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=icpx -D CMAKE_CXX_FLAGS="-fsycl -fsycl-device-code-split=per_kernel"
```

Note: `per_kernel` mode facilitates cases where the device lacks support for computations on a particular data type, e.g. double precision.

If you are building under Windows/DPC++ try:
```
cmake ..\mixbench-sycl -T "Intel(R) oneAPI DPC++ Compiler"  -D CMAKE_CXX_COMPILER=dpcpp -D CMAKE_CXX_FLAGS="-fsycl-device-code-split=per_kernel /EHsc"
```
Note: Adjust the platform toolset argument (*"Intel(R) oneAPI DPC++ Compiler"*) to whatever required, e.g. *"Intel(R) oneAPI DPC++ Compiler 2022"* for DPC++ 2022.1.

### AMD GPU via hipSYCL/ROCm

Here building for two device architectures (*gfx803* & *gfx1012*):

```
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=syclcc -D CMAKE_CXX_FLAGS="--hipsycl-targets='omp;hip:gfx803,gfx1012' -O2"
```
Note: Older ROCm releases might require adding `--rocm-device-lib-path=/opt/rocm/amdgcn/bitcode` to CMAKE_CXX_FLAGS

### NVidia clang/DPCPP
```
cmake ../mixbench-sycl -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_CXX_FLAGS="-fsycl -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda"
```

## Execution

In order to select the desired device to run the benchmark on, do pass the `-h` parameter
so the available devices are listed:

```
$ ./mixbench-sycl -h
mixbench-sycl (v0.04-3-g664f025)
Usage: mixbench-sycl [options] [device index [workgroup size [array size(1024^2)]]]

Options:
  -h or --help              Show this message
  -t or --use-os-timer      Use standard OS timer instead of SYCL profiling timer
Available SYCL devices:
  1. Intel(R) FPGA Emulation Device/Intel(R) FPGA Emulation Platform for OpenCL(TM)
  2. Intel(R) Core(TM) i3-8109U CPU @ 3.00GHz/Intel(R) OpenCL
  3. Intel(R) Iris(R) Plus Graphics 655 [0x3ea5]/Intel(R) OpenCL HD Graphics
  4. Intel(R) Core(TM) i3-8109U CPU @ 3.00GHz/Intel(R) OpenCL
  5. Intel(R) Iris(R) Plus Graphics 655 [0x3ea5]/Intel(R) Level-Zero
```

... and then pass the device number as the argument:

```
$ ./mixbench-sycl 5
mixbench-sycl (v0.04-3-g664f025)
Use "-h" argument to see available options
------------------------ Device specifications ------------------------
Platform:            Intel(R) Level-Zero
Device:              Intel(R) Iris(R) Plus Graphics 655 [0x3ea5]/Intel(R) Corporation
Driver version:      1.2.21270
Address bits:        64
GPU clock rate:      0 MHz
Total global mem:    12690 MB
Max allowed buffer:  4095 MB
SYCL version:        1.1
Total CUs:           48
-----------------------------------------------------------------------
Total GPU memory:     13307101184
Buffer size:          256MB
Elements per thread:  8
Thread fusion degree: 4
Timer:                SYCL event based
----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------
Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Half precision ops,,,,                Integer operations,,,
Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec
            0,      0.250,    5.29,    6.34,  25.36,      0.125,    9.63,    3.49,  27.88,      0.500,    5.04,   13.32,  26.64,     0.250,    5.23,    6.42,  25.68
            1,      0.750,    5.10,   19.74,  26.32,      0.375,    9.49,   10.60,  28.27,      1.500,    5.00,   40.25,  26.83,     0.750,    5.18,   19.43,  25.90
            2,      1.250,    5.29,   31.73,  25.38,      0.625,    9.78,   17.16,  27.45,      2.500,    4.62,   72.59,  29.03,     1.250,    5.21,   32.23,  25.78
            3,      1.750,    5.27,   44.55,  25.46,      0.875,    9.72,   24.15,  27.61,      3.500,    5.12,   91.78,  26.22,     1.750,    5.01,   46.87,  26.78
            4,      2.250,    5.26,   57.45,  25.53,      1.125,    9.09,   33.21,  29.52,      4.500,    4.62,  130.62,  29.03,     2.250,    4.61,   65.57,  29.14
            5,      2.750,    5.25,   70.35,  25.58,      1.375,    9.11,   40.51,  29.46,      5.500,    5.26,  140.31,  25.51,     2.750,    4.90,   75.28,  27.37
            6,      3.250,    5.06,   86.15,  26.51,      1.625,    9.09,   48.01,  29.54,      6.500,    4.62,  188.85,  29.05,     3.250,    4.86,   89.79,  27.63
            7,      3.750,    4.53,  111.09,  29.62,      1.875,    9.63,   52.27,  27.88,      7.500,    4.65,  216.63,  28.88,     3.750,    4.70,  107.11,  28.56
            8,      4.250,    4.74,  120.33,  28.31,      2.125,    9.37,   60.91,  28.66,      8.500,    4.60,  247.78,  29.15,     4.250,    4.55,  125.23,  29.47
            9,      4.750,    4.38,  145.42,  30.61,      2.375,    9.23,   69.07,  29.08,      9.500,    4.76,  268.08,  28.22,     4.750,    4.43,  143.86,  30.29
           10,      5.250,    4.73,  149.09,  28.40,      2.625,    9.31,   75.67,  28.83,     10.500,    4.55,  309.77,  29.50,     5.250,    4.42,  159.29,  30.34
           11,      5.750,    4.59,  168.17,  29.25,      2.875,    9.11,   84.69,  29.46,     11.500,    4.75,  324.78,  28.24,     5.750,    4.46,  173.02,  30.09
           12,      6.250,    4.39,  190.94,  30.55,      3.125,    8.90,   94.21,  30.15,     12.500,    4.40,  381.52,  30.52,     6.250,    4.50,  186.48,  29.84
           13,      6.750,    4.38,  206.79,  30.64,      3.375,    9.00,  100.67,  29.83,     13.500,    4.45,  406.98,  30.15,     6.750,    4.54,  199.70,  29.59
           14,      7.250,    4.41,  220.72,  30.44,      3.625,    9.00,  108.09,  29.82,     14.500,    4.41,  441.06,  30.42,     7.250,    4.56,  213.21,  29.41
           15,      7.750,    4.38,  237.52,  30.65,      3.875,    9.02,  115.35,  29.77,     15.500,    4.74,  439.27,  28.34,     7.750,    4.70,  221.36,  28.56
           16,      8.250,    4.39,  252.08,  30.55,      4.125,    9.03,  122.61,  29.72,     16.500,    4.43,  499.64,  30.28,     8.250,    4.89,  226.35,  27.44
           17,      8.750,    4.38,  268.11,  30.64,      4.375,    9.09,  129.14,  29.52,     17.500,    4.82,  487.16,  27.84,     8.750,    5.34,  219.77,  25.12
           18,      9.250,    4.39,  282.65,  30.56,      4.625,    9.09,  136.54,  29.52,     18.500,    4.44,  559.87,  30.26,     9.250,    5.41,  229.49,  24.81
           20,     10.250,    4.38,  314.04,  30.64,      5.125,    9.38,  146.63,  28.61,     20.500,    4.45,  617.83,  30.14,    10.250,    5.93,  231.87,  22.62
           22,     11.250,    4.36,  345.92,  30.75,      5.625,    9.16,  164.93,  29.32,     22.500,    4.45,  679.14,  30.18,    11.250,    6.46,  233.80,  20.78
           24,     12.250,    4.35,  377.65,  30.83,      6.125,    9.85,  166.94,  27.26,     24.500,    4.46,  737.88,  30.12,    12.250,    6.98,  235.42,  19.22
           28,     14.250,    4.37,  437.21,  30.68,      7.125,   11.26,  169.93,  23.85,     28.500,    4.48,  854.50,  29.98,    14.250,    8.05,  237.58,  16.67
           32,     16.250,    4.36,  499.84,  30.76,      8.125,   12.77,  170.75,  21.02,     32.500,    4.50,  968.97,  29.81,    16.250,    9.12,  239.18,  14.72
           40,     20.250,    4.41,  616.54,  30.45,     10.125,   15.88,  171.17,  16.91,     40.500,    5.02, 1083.80,  26.76,    20.250,   11.23,  242.07,  11.95
           48,     24.250,    4.81,  676.04,  27.88,     12.125,   18.48,  176.11,  14.52,     48.500,    5.91, 1101.80,  22.72,    24.250,   13.37,  243.40,  10.04
           56,     28.250,    5.48,  691.78,  24.49,     14.125,   21.38,  177.33,  12.55,     56.500,    6.34, 1195.28,  21.16,    28.250,   15.52,  244.32,   8.65
           64,     32.250,    6.17,  701.22,  21.74,     16.125,   24.28,  178.27,  11.06,     64.500,    6.99, 1238.30,  19.20,    32.250,   17.66,  245.05,   7.60
           80,     40.250,    7.59,  712.18,  17.69,     20.125,   30.55,  176.85,   8.79,     80.500,    8.37, 1290.93,  16.04,    40.250,   22.45,  240.58,   5.98
           96,     48.250,    8.99,  720.68,  14.94,     24.125,   35.85,  180.66,   7.49,     96.500,    9.78, 1323.72,  13.72,    48.250,   26.77,  241.96,   5.01
          128,     64.250,   11.84,  728.22,  11.33,     32.125,   47.47,  181.66,   5.65,    128.500,   12.62, 1366.48,  10.63,    64.250,   35.32,  244.14,   3.80
          192,     96.250,   19.89,  649.40,   6.75,     48.125,   71.77,  180.00,   3.74,    192.500,   18.37, 1406.37,   7.31,    96.250,   52.49,  246.11,   2.56
          256,    128.250,   25.72,  669.32,   5.22,     64.125,   94.85,  181.48,   2.83,    256.500,   32.84, 1048.44,   4.09,   128.250,   69.62,  247.23,   1.93
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
