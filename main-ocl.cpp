/**
 * main-ocl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include "timestamp.h"
#include <string.h>
#include "loclutil.h"
#include "mix_kernels_ocl.h"

#define VECTOR_SIZE (32*1024*1024)
//#define VECTOR_SIZE (16*1024*1024)

void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++)
		v[i] = i;
}

int main(int argc, char* argv[]) {
	printf("mixbench-ocl (compute & memory balancing GPU microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(double);

	cl_device_id dev_id = GetDeviceID();
	StoreDeviceInfo(dev_id, stdout);

	printf("Buffer size: %dMB\n", datasize/(1024*1024));
	
	double *c;
	c = (double*)malloc(datasize);
	init_vector(c, VECTOR_SIZE);

	mixbenchGPU(dev_id, c, VECTOR_SIZE, true);

	free(c);

	return 0;
}
