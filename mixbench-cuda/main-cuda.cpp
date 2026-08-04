/**
 * main-cuda.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"
#include "version_info.h"

#define VECTOR_SIZE (32 * 1024 * 1024)

void print_usage(const char* program_name) {
    printf("Usage: %s [GPU_ID]\n", program_name);
    printf("  GPU_ID    GPU to use (default: 0)\n");
    printf("  -h, --help  Show this message and list available CUDA GPUs\n");
}

void print_available_devices() {
    int device_count;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));

    printf("Available CUDA devices:\n");
    for (int device_id = 0; device_id < device_count; ++device_id) {
        cudaDeviceProp device_prop;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_prop, device_id));
        printf("  %d. %s\n", device_id, device_prop.name);
    }
}

int main(int argc, char* argv[]) {
    printf("mixbench (%s)\n", VERSION_INFO);

    int gpu_id = 0;
    if (argc > 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (argc == 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            printf("\n");
            print_available_devices();
            return 0;
        }

        char* end;
        long value;

        errno = 0;
        value = strtol(argv[1], &end, 10);
        if (errno != 0 || *argv[1] == '\0' || *end != '\0' || value < 0 || value > INT_MAX) {
            fprintf(stderr, "Error: GPU ID must be a non-negative integer.\n");
            print_usage(argv[0]);
            return 1;
        }
        gpu_id = (int)value;
    }

    int device_count;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
    if (gpu_id >= device_count) {
        fprintf(stderr, "Error: GPU ID %d is out of range; %d CUDA GPU%s available.\n",
                gpu_id, device_count, device_count == 1 ? " is" : "s are");
        print_usage(argv[0]);
        printf("\n");
        print_available_devices();
        return 1;
    }

    unsigned int datasize = VECTOR_SIZE * sizeof(double);

    CUDA_SAFE_CALL(cudaSetDevice(gpu_id));
    StoreDeviceInfo(stdout);

    size_t freeCUDAMem, totalCUDAMem;
    cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
    printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
    printf("Buffer size:          %dMB\n", datasize / (1024 * 1024));

    double* c;
    c = (double*)malloc(datasize);

    mixbenchGPU(c, VECTOR_SIZE);

    free(c);

    return 0;
}
