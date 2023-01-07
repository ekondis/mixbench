# mixbench-cpu

This is the OpenMP implementation of mixbench, targeted to CPUs.
Theoretically, it could also target GPU accelerators but it has been developed
with the CPUs in mind.
In particular, it has been tailored for GCC compiler (see below for more info).

## Running in docker

The easiest way to run CPU version is by docker:
`docker run --rm elkondis/mixbench-cpu`

This docker image re-compiles by tuning on your CPU architecture and executes the
benchmark.

## Notes

`mixbench-cpu` has been developed with `g++` (`gcc`) in mind.
As such, it has been validated on the particular compiler that it vectorizes and properly
unrolls the vectorized instructions as intended, in order to approach peak performance.
`clang` on the other hand, at the time of development, has been observed that it does not
properly produce optimum machine instruction sequences.
The nature of computations for loop iteration in this benchmark is inherently sequential.
So, it is essential that the compiler adequatelly unrolls the loop in the generated code
so the CPU does not stall due to dependencies.

## Building notes

The proper flags passed to the compiler (`-fopenmp -march=native -funroll-loops`) is taken care
by the CMakeLists script.
Thus, a simple cmake build invocation should be enough.
