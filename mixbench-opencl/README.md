# mixbench-opencl

This is the OpenCL implementation of mixbench.

## Building notes

Occasionally, (depending on the CMake version) the OpenCL files might not be
discovered automatically.
In such cases you might need to provide the OpenCL directories explicitly,
as seen in the examples below:

```
cmake ../mixbench-opencl -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/
cmake ../mixbench-opencl -DOpenCL_LIBRARY=/opt/rocm/lib/libOpenCL.so -DOpenCL_INCLUDE_DIR=/opt/rocm/opencl/include/
cmake ../mixbench-opencl -DOpenCL_LIBRARY=/opt/amdgpu-pro/lib/x86_64-linux-gnu/libOpenCL.so
```
