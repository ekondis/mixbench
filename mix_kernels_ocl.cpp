/**
 * mix_kernels_ocl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <cstdio>
#include <cstdarg>
#include <cstring>
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include "loclutil.h"

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define REGBLOCK_SIZE (8)

#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)

#if defined(_MSC_VER)
#define SIZE_T_FORMAT "%lu"
#else
#define SIZE_T_FORMAT "%zu"
#endif

enum KrnDataType{ kdt_int, kdt_float, kdt_double };

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

void flushed_printf(const char* format, ...){
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
}

void show_progress_init(int length){
	flushed_printf("[");
	for(int i=0; i<length; i++)
		flushed_printf(" ");
	flushed_printf("]");
	for(int i=0; i<=length; i++)
		flushed_printf("\b");
}

void show_progress_step(int domove, char newchar){
	flushed_printf("%c", newchar);
	if( !domove )
		flushed_printf("\b");
}

void show_progress_done(void){
	flushed_printf("\n");
}

double get_event_duration(cl_event ev){
	cl_ulong ev_t_start, ev_t_finish;
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_t_start, NULL) );
	OCL_SAFE_CALL( clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_t_finish, NULL) );
	double time = (ev_t_finish-ev_t_start)/1000000.0;
	return time;
}

cl_kernel BuildKernel(cl_context context, cl_device_id dev_id, const char *source, const char *parameters){
	cl_int errno;
	const char **sources = &source;
	cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &errno);
	OCL_SAFE_CALL(errno);
	errno = clBuildProgram(program, 1, &dev_id, parameters, NULL, NULL);
	if( errno != CL_SUCCESS ){
		fprintf(stderr, "Program built error code: %d\n", errno);
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
	// Kernel creation
	cl_kernel kernel = clCreateKernel(program, "benchmark_func", &errno);
	OCL_SAFE_CALL(errno);
	return kernel;
}

void ReleaseKernelNProgram(cl_kernel kernel){
	cl_program program_tmp;
	OCL_SAFE_CALL( clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(program_tmp), &program_tmp, NULL) );
	OCL_SAFE_CALL( clReleaseKernel(kernel) );
	OCL_SAFE_CALL( clReleaseProgram(program_tmp) );
}

void runbench_warmup(cl_command_queue queue, cl_kernel kernel, cl_mem cbuffer, long size, size_t workgroupsize){
	const long reduced_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/32;

	const size_t dimBlock[1] = {workgroupsize};
	const size_t dimReducedGrid[1] = {(size_t)reduced_grid_size};

	const short seed = 1;
	OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_short), &seed) );
	OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &cbuffer) );

	OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, dimReducedGrid, dimBlock, 0, NULL, NULL) );
}

void runbench(int memory_ratio, cl_command_queue queue, cl_kernel kernels[kdt_double+1][32+1], cl_mem cbuffer, long size, size_t workgroupsize){
	if( memory_ratio>UNROLL_ITERATIONS ){
		fprintf(stderr, "ERROR: memory_ratio exceeds UNROLL_ITERATIONS\n");
		exit(1);
	}

	const long compute_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/2;
	
	const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*compute_grid_size;
	const long long memoryoperations = (long long)(COMP_ITERATIONS)*compute_grid_size;

	const size_t dimBlock[1] = {workgroupsize};
	const size_t dimGrid[1] = {(size_t)compute_grid_size};

	cl_event event;
	
	const short seed_f = 1.0f;
	cl_kernel kernel = kernels[kdt_float][memory_ratio];
	OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_float), &seed_f) );
	OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &cbuffer) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, dimGrid, dimBlock, 0, NULL, &event) );
	OCL_SAFE_CALL( clWaitForEvents(1, &event) );
	double kernel_time_mad_sp = get_event_duration(event);
	OCL_SAFE_CALL( clReleaseEvent( event ) );

	const short seed_d = 1.0;
	double kernel_time_mad_dp;
	kernel = kernels[kdt_double][memory_ratio];
	if( kernel ){
		OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_double), &seed_d) );
		OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &cbuffer) );
		OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, dimGrid, dimBlock, 0, NULL, &event) );
		OCL_SAFE_CALL( clWaitForEvents(1, &event) );
		kernel_time_mad_dp = get_event_duration(event);
		OCL_SAFE_CALL( clReleaseEvent( event ) );
	} else 
		kernel_time_mad_dp = 0.0;

	const short seed_i = 1.0;
	kernel = kernels[kdt_int][memory_ratio];
	OCL_SAFE_CALL( clSetKernelArg(kernel, 0, sizeof(cl_int), &seed_i) );
	OCL_SAFE_CALL( clSetKernelArg(kernel, 1, sizeof(cl_mem), &cbuffer) );
	OCL_SAFE_CALL( clEnqueueNDRangeKernel(queue, kernel, 1, NULL, dimGrid, dimBlock, 0, NULL, &event) );
	OCL_SAFE_CALL( clWaitForEvents(1, &event) );
	double kernel_time_mad_int = get_event_duration(event);
	OCL_SAFE_CALL( clReleaseEvent( event ) );

	const double memaccesses_ratio = (double)(memory_ratio)/UNROLL_ITERATIONS;
	const double computations_ratio = 1.0-memaccesses_ratio;

	printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  %8.3f,%8.2f,%8.2f,%7.2f\n", 
		UNROLL_ITERATIONS-memory_ratio,
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(float)),
		kernel_time_mad_sp,
		(computations_ratio*(double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(double)),
		kernel_time_mad_dp,
		(computations_ratio*(double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(int)),
		kernel_time_mad_int,
		(computations_ratio*(double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );
}

extern "C" void mixbenchGPU(cl_device_id dev_id, double *c, long size, bool block_strided, bool host_allocated, size_t workgroupsize){
	const char *benchtype;
	if(block_strided)
		benchtype = "Workgroup";
	else
		benchtype = "NDRange";
	printf("Workitem stride       : %s\n", benchtype);
	const char *buffer_allocation = host_allocated ? "Host allocated" : "Device allocated";
	printf("Buffer allocation     : %s\n", buffer_allocation);

	// Set context properties
	cl_platform_id p_id;
	OCL_SAFE_CALL( clGetDeviceInfo(dev_id, CL_DEVICE_PLATFORM, sizeof(p_id), &p_id, NULL) );
	size_t length;
	OCL_SAFE_CALL( clGetDeviceInfo(dev_id, CL_DEVICE_EXTENSIONS, 0, NULL, &length) );
	char *extensions = (char*)alloca(length);
	OCL_SAFE_CALL( clGetDeviceInfo(dev_id, CL_DEVICE_EXTENSIONS, length, extensions, NULL) );
	bool enable_dp = strstr(extensions, "cl_khr_fp64") != NULL;

	cl_context_properties ctxProps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)p_id, 0 };

	cl_int errno;
	// Create context
	cl_context context = clCreateContext(ctxProps, 1, &dev_id, NULL, NULL, &errno);
	OCL_SAFE_CALL(errno);

	cl_mem_flags buf_flags = CL_MEM_READ_WRITE;
        if( host_allocated )
                buf_flags |= CL_MEM_ALLOC_HOST_PTR;
	cl_mem c_buffer = clCreateBuffer(context, buf_flags, size*sizeof(double), NULL, &errno);
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

	// Load source, create program and all kernels
	printf("Loading kernel source file...\n");
	const char c_param_format_str[] = "-cl-std=CL1.1 -cl-mad-enable -Dclass_T=%s -Dblockdim=" SIZE_T_FORMAT " -Dmemory_ratio=%d %s %s";
	const char *c_empty = "";
	const char *c_striding = block_strided ? "-DBLOCK_STRIDED" : c_empty;
	const char *c_enable_dp = "-DENABLE_DP";
	char c_build_params[256];
	const char *c_kernel_source = {ReadFile("mix_kernels.cl")};
	printf("Precompilation of kernels... ");
	sprintf(c_build_params, c_param_format_str, "short", workgroupsize, 0, c_striding, c_empty);

	cl_kernel kernel_warmup = BuildKernel(context, dev_id, c_kernel_source, c_build_params);

	show_progress_init(32+1);
	cl_kernel kernels[kdt_double+1][32+1];
	for(int i=0; i<=32; i++){
		show_progress_step(0, '\\');
		sprintf(c_build_params, c_param_format_str, "float", workgroupsize, i, c_striding, c_empty);
		//printf("%s\n",c_build_params);
		kernels[kdt_float][i] = BuildKernel(context, dev_id, c_kernel_source, c_build_params);

		show_progress_step(0, '|');
		sprintf(c_build_params, c_param_format_str, "int", workgroupsize, i, c_striding, c_empty);
		//printf("%s\n",c_build_params);
		kernels[kdt_int][i] = BuildKernel(context, dev_id, c_kernel_source, c_build_params);

		if( enable_dp ){
			show_progress_step(0, '/');
			sprintf(c_build_params, c_param_format_str, "double", workgroupsize, i, c_striding, c_enable_dp);
			//printf("%s\n",c_build_params);
			kernels[kdt_double][i] = BuildKernel(context, dev_id, c_kernel_source, c_build_params);
		} else
			kernels[kdt_double][i] = 0;
		show_progress_step(1, '>');
	}
	show_progress_done();
	free((char*)c_kernel_source);

	runbench_warmup(cmd_queue, kernel_warmup, c_buffer, size, workgroupsize);

	// Synchronize in order to wait for memory operations to finish
	OCL_SAFE_CALL( clFinish(cmd_queue) );

	printf("---------------------------------------------------------- CSV data ----------------------------------------------------------\n");
	printf("Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Integer operations,,, \n");
	printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec\n");

	for(int i=32; i>=0; i--)
		runbench(i, cmd_queue, kernels, c_buffer, size, workgroupsize);

	printf("------------------------------------------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	OCL_SAFE_CALL( clEnqueueReadBuffer(cmd_queue, c_buffer, CL_TRUE, 0, size*sizeof(double), c, 0, NULL, NULL) );

	// Release kernels and program
	ReleaseKernelNProgram(kernel_warmup);
	for(int i=0; i<=32; i++){
		ReleaseKernelNProgram(kernels[kdt_float][i]);
		ReleaseKernelNProgram(kernels[kdt_int][i]);
		if( enable_dp )
			ReleaseKernelNProgram(kernels[kdt_double][i]);
	}

	// Release buffer
	OCL_SAFE_CALL( clReleaseMemObject(c_buffer) );
}
