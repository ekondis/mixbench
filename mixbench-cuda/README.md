# mixbench-cuda

This is the CUDA implementation of mixbench.
It is actually the original implementation of this benchmark.

## Building

To build the executable, run the following commands.

> The minimum required CMake version is 3.18.

```sh
mkdir build
cd build
cmake ../mixbench-cuda -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build ./
```

This will build and write a `mixbench-cuda` executable file in the `build/`
directory, compiled with support for the native CUDA architecture. Note that
the `-arch=native` flag was [introduced in CUDA 11.5 update 1][1]. If you
are using a prior version, or wish to compile the program for a specific
architecture, replace `native` in the above command with the architecture.
For example, to compile for the `sm_120` architecture, we would run:

```
mkdir build
cd build
cmake ../mixbench-cuda -DCMAKE_CUDA_ARCHITECTURES=sm_120
cmake --build ./
```

[1]: https://docs.nvidia.com/cuda/cuda-features-archive/index.html#compiler
