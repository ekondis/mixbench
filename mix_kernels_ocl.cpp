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

#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)

char* ReadFile(const char *filename){
	char *buffer = NULL;
	int file_size, read_size;
	FILE *file = fopen(filename, "r");
	if(!file)
		return NULL;
	// Seek EOF
	fseek(file, 0, SEEK_END);
	// Get offset
	file_size = ftell(file);
	rewind(file);
	buffer = (char*)malloc(sizeof(char) * (file_size+1));
	read_size = fread(buffer, sizeof(char), file_size, file);
	buffer[file_size] = '\0';
	if(file_size != read_size) {
		free(buffer);
		buffer = NULL;
	}
	return buffer;
}

/*void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}*/

/*void runbench_warmup(double *cd, long size){
	const long reduced_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/32;
	const int BLOCK_SIZE = 256;
	const int shared_size = 0;

	const size_t dimBlock[1] = {BLOCK_SIZE};
	const size_t dimReducedGrid[1] = {reduced_grid_size};

	benchmark_func< short, BLOCK_SIZE, 0, 0 ><<< dimReducedGrid, dimBlock, shared_size >>>((short)1, (short*)cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}*/

/*template<int memory_ratio>
void runbench(double *cd, long size){
	if( memory_ratio>UNROLL_ITERATIONS ){
		fprintf(stderr, "ERROR: memory_ratio exceeds UNROLL_ITERATIONS\n");
		exit(1);
	}
		
	const long compute_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/2;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*compute_grid_size;
	const long long memoryoperations = (long long)(COMP_ITERATIONS)*compute_grid_size;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;
	const int shared_count = 0;

	initializeEvents(&start, &stop);
	benchmark_func< float, BLOCK_SIZE, memory_ratio, 0 ><<< dimGrid, dimBlock, shared_count*sizeof(float) >>>(1.0f, (float*)cd);
	float kernel_time_mad_sp = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_func< double, BLOCK_SIZE, memory_ratio, 0 ><<< dimGrid, dimBlock, shared_count*sizeof(double) >>>(1.0, cd);
	float kernel_time_mad_dp = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_func< int, BLOCK_SIZE, memory_ratio, 0 ><<< dimGrid, dimBlock, shared_count*sizeof(int) >>>(1, (int*)cd);
	float kernel_time_mad_int = finalizeEvents(start, stop);

	const double memaccesses_ratio = (double)(memory_ratio)/UNROLL_ITERATIONS;
	const double computations_ratio = 1.0-memaccesses_ratio;

	printf("      %2d/%2d,     %8.2f,%8.2f,%7.2f,%8.2f,%8.2f,%7.2f,%8.2f,%8.2f,%7.2f\n", 
		UNROLL_ITERATIONS-memory_ratio, memory_ratio,
		kernel_time_mad_sp,
		(computations_ratio*(double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
		kernel_time_mad_dp,
		(computations_ratio*(double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
		kernel_time_mad_int,
		(computations_ratio*(double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );
}*/

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

	// Load kernels
	const char *c_kernel_source[1] = {ReadFile("mix_kernels.cl")};
	puts(c_kernel_source[0]);

	// Create program and all kernels
	const char *build_options = "";
	cl_program program = clCreateProgramWithSource(context, 1, c_kernel_source, NULL, &errno);
	OCL_SAFE_CALL(errno);
	if( clBuildProgram(program, 1, &dev_id, build_options, NULL, NULL) != CL_SUCCESS ){
		size_t log_size;
		OCL_SAFE_CALL( clGetProgramBuildInfo(program, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );
		char *log = (char*)alloca(log_size);
		OCL_SAFE_CALL( clGetProgramBuildInfo(program, dev_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL) );
		OCL_SAFE_CALL( clReleaseProgram(program) );
		fprintf(stderr, "------------------------------------ Kernel compilation log ----------------------------------\n");
		fprintf(stderr, "%s", log);
		fprintf(stderr, "----------------------------------------------------------------------------------------------\n");
		exit(EXIT_FAILURE);
	}

	// Synchronize in order to wait for memory operations to finish
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	printf("----------------------------------------- EXCEL data -----------------------------------------\n");
	printf("Operations ratio,  Single Precision ops,,,   Double precision ops,,,     Integer operations,, \n");
	printf("  compute/memory,    Time,  GFLOPS, GB/sec,    Time,  GFLOPS, GB/sec,    Time,   GIOPS, GB/sec\n");

//	runbench_warmup(cd, size);

/*	runbench<32>(cd, size);
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

	// Release kernels and program
	OCL_SAFE_CALL( clReleaseProgram(program) );
	free((char*)c_kernel_source[0]);

	// Release buffer
	OCL_SAFE_CALL( clReleaseMemObject(c_buffer) );
}
