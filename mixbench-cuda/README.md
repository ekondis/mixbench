# mixbench-cuda

This is the CUDA implementation of mixbench.
It is actually the original implementation of this benchmark.

## Building locally

To build the executable, run the following commands.

```sh
mkdir build
cd build
cmake ../mixbench-cuda -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ./
```

This will build and write a `mixbench-cuda` executable file in the `build/`
directory, compiled with support for the native CUDA architecture. Note that
the `-arch=native` flag was [introduced in CUDA 11.5 update 1][1]. If you
are using a prior version, follow the below instructions to compile for a given
architecture.

## Building for specific architectures

If you wish to compile the program for specific architectures, i.e., `sm_120`
and `sm_52`, edit `CMakeLists.txt` in this directory and change the following
line as shown below.

```diff
--- a/mixbench-cuda/CMakeLists.txt
+++ b/mixbench-cuda/CMakeLists.txt
@@ -7,7 +7,7 @@ include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
 string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
 string(APPEND CMAKE_CUDA_FLAGS " -Wno-deprecated-gpu-targets")
 string(APPEND CMAKE_CUDA_FLAGS " --cudart=static")
-string(APPEND CMAKE_CUDA_FLAGS " -arch=native")
+string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_120,code=[sm_120,compute_120] -gencode arch=compute_52,code=compute_52")

 # Get version info from git tag
 execute_process(COMMAND git describe --tags --always
```

[1]: https://docs.nvidia.com/cuda/cuda-features-archive/index.html#compiler
