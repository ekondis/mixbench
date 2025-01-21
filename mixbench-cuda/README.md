# mixbench-cuda

This is the CUDA implementation of mixbench.
It is actually the original implementation of this benchmark.

## Building notes

Building should be straightforward by using the respective `CMakeList.txt` file.


## Usage

Use `--gpu` option to select which GPU to benchmark (otherwise defaults to 0), e.g.:
```bash
./mixbench-cuda --gpu 2
```