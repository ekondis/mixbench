/**
 * main-ocl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "loclutil.h"
#include "mix_kernels_ocl.h"

#define DEF_VECTOR_SIZE (32*1024*1024)

typedef struct{
	int device_index;
	bool block_strided;
	bool host_allocated;
	int wg_size;
	unsigned int vecwidth;
} ArgParams;

void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++)
		v[i] = i;
}

// Argument parsing
// returns whether program execution should continue (true) or just print help output (false)
bool argument_parsing(int argc, char* argv[], ArgParams *output){
	int arg_count = 0;
	for(int i=1; i<argc; i++) {
		if( (strcmp(argv[i], "-h")==0) || (strcmp(argv[i], "--help")==0) ) {
			return false;
		} else if( (strcmp(argv[i], "-w")==0) || (strcmp(argv[i], "--workgroup-stride")==0) ) {
			output->block_strided = true;
		} else if( (strcmp(argv[i], "-H")==0) || (strcmp(argv[i], "--host-alloc")==0) ) {
			output->host_allocated = true;
		} else {
			unsigned long value = strtoul(argv[i], NULL, 10);
			switch( arg_count ){
				// device selection
				case 0:
					output->device_index = value;
					arg_count++;
					break;
				// workgroup size
				case 1:
					output->wg_size = value;
					arg_count++;
					break;
				// array size (x1024^2)
				case 2:
					output->vecwidth = value;
					arg_count++;
					break;
				default:
					return false;
			}
		}
	}
	return true;
}

int main(int argc, char* argv[]) {
	printf("mixbench-ocl (compute & memory balancing GPU microbenchmark)\n");

	ArgParams args = {1, false, false, 256, DEF_VECTOR_SIZE/(1024*1024)};
	if( !argument_parsing(argc, argv, &args) ){
		printf("Usage: mixbench-ocl [options] [device index [workgroup size [array size(1024^2)]]]\n");
		printf("\nOptions:\n"
			"  -h or --help              Show this message\n"
			"  -H or --host-alloc        Use host allocated buffer (CL_MEM_ALLOC_HOST_PTR)\n"
			"  -w or --workgroup-stride  Workitem strides equal to the width of a workgroup length (default: NDRange length)\n"
			"\n");

		GetDeviceID(0, stdout);
		exit(1);
	}

	printf("Use \"-h\" argument to see available options\n");
	
	const size_t VEC_WIDTH = 1024*1024*args.vecwidth;
	unsigned int datasize = VEC_WIDTH*sizeof(double);

	cl_device_id dev_id = GetDeviceID(args.device_index, NULL);

	if( dev_id == NULL ){
		fprintf(stderr, "Error: No OpenCL device selected\n");
		exit(1);
	}
	StoreDeviceInfo(dev_id, stdout);

	printf("Buffer size: %dMB\n", datasize/(1024*1024));
	printf("Workgroup size: %d\n", args.wg_size);
	
	double *c;
	c = (double*)malloc(datasize);
	init_vector(c, VEC_WIDTH);

	mixbenchGPU(dev_id, c, VEC_WIDTH, args.block_strided, args.host_allocated, args.wg_size);

	free(c);

	return 0;
}
