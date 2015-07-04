# mixbench
The purpose of this benchmark tool is to evaluate performance bounds of GPUs on mixed operational intensity kernels. The executed kernel is customized on a range of different operational intensity values. Modern GPUs are able to hide memory latency by switching execution to compute operations. Using this tool one can assess the practical optimum balance in both types of operations for a GPU. It's is based on CUDA programming platform so it can be executed only on NVidia GPUs. A long term goal is to develop an OpenCL port.

Kernel types
--------------

Three types of experiments are executed combined with global memory accesses:

1. Single precision FLOPs (multiply-additions)
2. Double precision FLOPs (multiply-additions)
3. Integer multiply-addition operations

Building program
--------------

In order to build the program you should make sure that the following variable in "Makefile" is set to the CUDA installation directory:

```
CUDA_INSTALL_PATH = /usr/local/cuda
```

Afterwards, just do make:

```
make
```

Two executables will be produced: "mixbench-cuda" & "mixbench-cuda-bs". The former applies grid strides between accesses of the same thread where the latter applies block size strides.

Execution results
--------------

A typical execution output on a GTX480 GPU is:
```
mixbench (compute & memory balancing GPU microbenchmark)
------------------------ Device specifications ------------------------
Device:              GeForce GTX 480
CUDA driver version: 5.50
GPU clock rate:      1401 MHz
Memory clock rate:   924 MHz
Memory bus width:    384 bits
WarpSize:            32
L2 cache size:       768 KB
Total global mem:    1535 MB
ECC enabled:         No
Compute Capability:  2.0
Total SPs:           480 (15 MPs x 32 SPs/MP)
Compute throughput:  1344.96 GFlops (theoretical single precision FMAs)
Memory bandwidth:    177.41 GB/sec
-----------------------------------------------------------------------
Total GPU memory 1610285056, free 1195106304
Buffer size: 256MB
Trade-off type:compute with global memory (block strided)
---- EXCEL data ----
Operations ratio ;  Single Precision ops ;;;  Double precision ops ;;;    Integer operations   
  compute/memory ;    Time;  GFLOPS; GB/sec;    Time;  GFLOPS; GB/sec;    Time;   GIOPS; GB/sec
       0/32      ; 240.531;    0.00; 142.85; 475.150;    0.00; 144.63; 240.205;    0.00; 143.04
       1/31      ; 233.548;    9.20; 142.52; 460.193;    4.67; 144.66; 233.484;    9.20; 142.56
       2/30      ; 225.249;   19.07; 143.01; 445.144;    9.65; 144.73; 225.235;   19.07; 143.02
       3/29      ; 218.552;   29.48; 142.48; 430.575;   14.96; 144.64; 218.745;   29.45; 142.35
       4/28      ; 210.345;   40.84; 142.93; 415.425;   20.68; 144.74; 210.091;   40.89; 143.10
       5/27      ; 203.132;   52.86; 142.72; 400.472;   26.81; 144.78; 203.275;   52.82; 142.62
       6/26      ; 194.468;   66.26; 143.56; 385.434;   33.43; 144.86; 194.314;   66.31; 143.67
       7/25      ; 187.470;   80.19; 143.19; 370.915;   40.53; 144.74; 187.475;   80.18; 143.18
       8/24      ; 175.115;   98.11; 147.16; 355.723;   48.30; 144.89; 175.132;   98.10; 147.14
       9/23      ; 171.760;  112.53; 143.78; 341.353;   56.62; 144.70; 171.920;  112.42; 143.65
      10/22      ; 163.397;  131.43; 144.57; 326.007;   65.87; 144.92; 163.252;  131.54; 144.70
      11/21      ; 155.797;  151.62; 144.73; 311.655;   75.80; 144.70; 155.814;  151.61; 144.71
      12/20      ; 146.573;  175.82; 146.51; 296.386;   86.95; 144.91; 146.662;  175.71; 146.42
      13/19      ; 138.853;  201.06; 146.93; 281.757;   99.08; 144.81; 138.941;  200.93; 146.83
      14/18      ; 129.727;  231.75; 148.98; 266.401;  112.86; 145.10; 129.744;  231.72; 148.97
      15/17      ; 121.228;  265.72; 150.57; 251.283;  128.19; 145.28; 121.339;  265.47; 150.43
      16/16      ; 120.065;  286.18; 143.09; 235.740;  145.75; 145.75; 120.122;  286.04; 143.02
      17/15      ; 111.357;  327.84; 144.64; 219.472;  166.34; 146.77; 111.528;  327.34; 144.41
      18/14      ; 106.430;  363.19; 141.24; 231.498;  166.98; 129.87; 106.541;  362.82; 141.10
      19/13      ;  96.118;  424.50; 145.22; 243.534;  167.54; 114.63;  96.494;  422.85; 144.66
      20/12      ;  89.602;  479.34; 143.80; 256.247;  167.61; 100.57;  89.642;  479.13; 143.74
      21/11      ;  81.976;  550.13; 144.08; 269.055;  167.61;  87.80;  83.091;  542.74; 142.15
      22/10      ;  76.066;  621.10; 141.16; 282.898;  167.00;  75.91;  76.068;  621.08; 141.15
      23/ 9      ;  65.631;  752.57; 147.24; 295.743;  167.01;  65.35;  76.895;  642.33; 125.67
      24/ 8      ;  60.809;  847.57; 141.26; 307.479;  167.62;  55.87;  80.099;  643.45; 107.24
      25/ 7      ;  52.032; 1031.82; 144.45; 321.449;  167.02;  46.76;  83.296;  644.53;  90.23
      26/ 6      ;  48.321; 1155.49; 133.33; 334.305;  167.02;  38.54;  86.519;  645.35;  74.46
      27/ 5      ;  49.519; 1170.90; 108.42; 347.157;  167.02;  30.93;  89.729;  646.19;  59.83
      28/ 4      ;  50.704; 1185.90;  84.71; 360.013;  167.02;  23.86;  92.891;  647.31;  46.24
      29/ 3      ;  52.024; 1197.09;  61.92; 372.867;  167.02;  17.28;  96.115;  647.94;  33.51
      30/ 2      ;  53.377; 1206.97;  40.23; 385.722;  167.02;  11.13;  99.328;  648.61;  21.62
      31/ 1      ;  53.437; 1245.80;  20.09; 397.203;  167.60;   5.41; 101.247;  657.52;  10.61
      32/ 0      ;  53.558; 1283.08;   0.00; 410.012;  167.60;   0.00; 102.494;  670.47;   0.00
--------------------
```
