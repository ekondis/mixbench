/**
 * mix_kernels_ocl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
//#include <math_constants.h>
#include "loclutil.h"
#include "timestamp.h"

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define REGBLOCK_SIZE (4)

/// .....

extern "C" void mixbenchGPU(cl_device_id dev_id, double *c, long size){
#ifdef BLOCK_STRIDED
	const char *benchtype = "compute with global memory (block strided)";
#else
	const char *benchtype = "compute with global memory (grid strided)";
#endif
	printf("Trade-off type:%s\n", benchtype);

	// Set context properties
	cl_platform_id p_id;
	OCL_SAFE_CALL( clGetDeviceInfo(dev_id, CL_DEVICE_PLATFORM, sizeof(p_id), &p_id,	NULL) );

	cl_context_properties ctxProps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)p_id, 0 };

	cl_int errno;
	// Create context
	cl_context context = clCreateContext(ctxProps, 1, &dev_id, NULL, NULL, &errno);
	OCL_SAFE_CALL(errno);

	cl_mem c_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size*sizeof(double), NULL, &errno);
	OCL_SAFE_CALL(errno);
	
	// Create command queue
	cl_command_queue cmd_queue = clCreateCommandQueue(context, dev_id, CL_QUEUE_PROFILING_ENABLE, &errno);
	OCL_SAFE_CALL(errno);

	// Set data on device memory
	cl_int *mapped_data = (cl_int*)clEnqueueMapBuffer(cmd_queue, c_buffer, CL_TRUE, CL_MAP_WRITE, 0, size*sizeof(double), 0, NULL, NULL, &errno);
	OCL_SAFE_CALL(errno);
	for(int i=0; i<size; i++)
		mapped_data[i] = 0;
	clEnqueueUnmapMemObject(cmd_queue, c_buffer, mapped_data, 0, NULL, NULL);

//	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	printf("----------------------------------------- EXCEL data -----------------------------------------\n");
	printf("Operations ratio,  Single Precision ops,,,   Double precision ops,,,     Integer operations,, \n");
	printf("  compute/memory,    Time,  GFLOPS, GB/sec,    Time,  GFLOPS, GB/sec,    Time,   GIOPS, GB/sec\n");

/*	runbench_warmup(cd, size);

	runbench<32>(cd, size);
	runbench<31>(cd, size);
	runbench<30>(cd, size);
	runbench<29>(cd, size);
	runbench<28>(cd, size);
	runbench<27>(cd, size);
	runbench<26>(cd, size);
	runbench<25>(cd, size);
	runbench<24>(cd, size);
	runbench<23>(cd, size);
	runbench<22>(cd, size);
	runbench<21>(cd, size);
	runbench<20>(cd, size);
	runbench<19>(cd, size);
	runbench<18>(cd, size);
	runbench<17>(cd, size);
	runbench<16>(cd, size);
	runbench<15>(cd, size);
	runbench<14>(cd, size);
	runbench<13>(cd, size);
	runbench<12>(cd, size);
	runbench<11>(cd, size);
	runbench<10>(cd, size);
	runbench<9>(cd, size);
	runbench<8>(cd, size);
	runbench<7>(cd, size);
	runbench<6>(cd, size);
	runbench<5>(cd, size);
	runbench<4>(cd, size);
	runbench<3>(cd, size);
	runbench<2>(cd, size);
	runbench<1>(cd, size);
	runbench<0>(cd, size);*/

	printf("----------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	OCL_SAFE_CALL( clEnqueueReadBuffer(cmd_queue, c_buffer, CL_TRUE, 0, size*sizeof(double), 	c, 	0, NULL, NULL) );
	//CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	// Release buffer
	OCL_SAFE_CALL( clReleaseMemObject(c_buffer) );
}
