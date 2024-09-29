/**
 * main-cuda.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"
#include "version_info.h"

#define VECTOR_SIZE (32 * 1024 * 1024)

void print_usage(const char* program_name) {
    printf("Usage: %s [--gpu <GPU_ID>]\n", program_name);
    printf("Options:\n");
    printf("  --gpu <GPU_ID>    Specify the GPU ID to use (default: 0)\n");
}

int main(int argc, char* argv[]) {
    printf("mixbench (%s)\n", VERSION_INFO);

    int gpu_id = 0;
    int i;
    for (i=1;i<argc;i++) {
        if(strcmp(argv[i], "--gpu") == 0) {
            if(i + 1 < argc) {
                gpu_id = atoi(argv[i + 1]);
                if(gpu_id < 0) {
                    fprintf(stderr, "Error: GPU ID must be a non-negative integer.\n");
                    print_usage(argv[0]);
                    return 1;
                }
                i++; // Skip the next argument as it's the GPU ID
            } else {
                fprintf(stderr, "Error: --gpu option requires an argument.\n");
                print_usage(argv[0]);
                return 1;
            }
        } else if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    unsigned int datasize = VECTOR_SIZE * sizeof(double);

    cudaSetDevice(gpu_id);
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
