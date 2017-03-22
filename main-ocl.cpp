/**
 * main-ocl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "loclutil.h"
#ifdef READONLY
#include "mix_kernels_ocl_ro.h"
#else
#include "mix_kernels_ocl.h"
#endif
#include "version_info.h"

#ifdef READONLY
#define DEF_VECTOR_SIZE (32*1024*1024)
#else
#define DEF_VECTOR_SIZE (8*1024*1024)
#endif

typedef struct{
	int device_index;
	bool block_strided;
	bool host_allocated;
	bool use_os_timer;
	int wg_size;
	unsigned int vecwidth;
#ifdef READONLY
	unsigned int elements_per_wi;
	unsigned int fusion_degree;
#endif
} ArgParams;

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
		} else if( (strcmp(argv[i], "-t")==0) || (strcmp(argv[i], "--use-os-timer")==0) ) {
			output->use_os_timer = true;
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
#ifdef READONLY
				// elements per workitem
				case 3:
					output->elements_per_wi = value;
					arg_count++;
					break;
				case 4:
					output->fusion_degree = value;
					arg_count++;
					break;
#endif
				default:
					return false;
			}
		}
	}
	return true;
}

int main(int argc, char* argv[]) {
#ifdef READONLY
	printf("mixbench-ocl/read-only (%s)\n", VERSION_INFO);
#else
	printf("mixbench-ocl/alternating (%s)\n", VERSION_INFO);
#endif

#ifdef READONLY
	ArgParams args = {1, false, false, false, 256, DEF_VECTOR_SIZE/(1024*1024), 8, 4};
#else
	ArgParams args = {1, false, false, false, 256, DEF_VECTOR_SIZE/(1024*1024)};
#endif

	if( !argument_parsing(argc, argv, &args) ){
#ifdef READONLY
		printf("Usage: mixbench-ocl [options] [device index [workgroup size [array size(1024^2) [elements per workitem [fusion degree]]]]]\n");
#else
		printf("Usage: mixbench-ocl [options] [device index [workgroup size [array size(1024^2)]]]\n");
#endif
		printf("\nOptions:\n"
			"  -h or --help              Show this message\n"
			"  -H or --host-alloc        Use host allocated buffer (CL_MEM_ALLOC_HOST_PTR)\n"
			"  -w or --workgroup-stride  Workitem strides equal to the width of a workgroup length (default: NDRange length)\n"
			"  -t or --use-os-timer      Use standard OS timer instead of OpenCL profiling timer\n"
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

	printf("Buffer size:            %dMB\n", datasize/(1024*1024));
	printf("Workgroup size:         %d\n", args.wg_size);
#ifdef READONLY
	printf("Elements per workitem:  %d\n", args.elements_per_wi);
	printf("Workitem fusion degree: %d\n", args.fusion_degree);
#endif
	// Check if selected workgroup size is supported
	if( GetMaxDeviceWGSize(dev_id)<(size_t)args.wg_size ){
		fprintf(stderr, "Error: Unsupported workgroup size (%u).\n", args.wg_size);
		exit(1);		
	}

	double *c;
	c = (double*)malloc(datasize);

#ifdef READONLY
	mixbenchGPU(dev_id, c, VEC_WIDTH, args.block_strided, args.host_allocated, args.use_os_timer, args.wg_size, args.elements_per_wi, args.fusion_degree);
#else
	mixbenchGPU(dev_id, c, VEC_WIDTH, args.block_strided, args.host_allocated, args.use_os_timer, args.wg_size);
#endif

	free(c);

	return 0;
}
